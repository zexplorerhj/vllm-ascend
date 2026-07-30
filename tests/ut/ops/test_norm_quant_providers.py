from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from norm_quant import (  # noqa: E402
    CudaNormQuantOperatorTest,
    NormQuantOperatorTestBase,
    NormQuantVariant,
    create_norm_quant_operator,
)
try:
    from norm_quant import NpuNormQuantOperatorTest  # noqa: E402
except ImportError:
    NpuNormQuantOperatorTest = None
from operator_test_framework import PrecisionType  # noqa: E402


VLLM_STATIC_RMS = "cuda_vllm_rms_norm_static_fp8_quant_out"
FLASHINFER_STATIC_RMS = "cuda_flashinfer_rmsnorm_quant_fp8_out_pdl"
VLLM_STATIC_ADD = (
    "cuda_vllm_fused_add_rms_norm_static_fp8_quant_out"
)
FLASHINFER_STATIC_ADD = (
    "cuda_flashinfer_fused_add_rmsnorm_quant_fp8_out_pdl"
)
VLLM_DYNAMIC_ADD = (
    "cuda_vllm_fused_add_rms_norm_dynamic_per_token_fp8_quant_out"
)


def _cpu_copy(value, *, device, dtype=None):
    del device
    return value.to(dtype=dtype or value.dtype).clone()


def _operator(variant):
    return CudaNormQuantOperatorTest(variant, PrecisionType.FP8)


class _CommonNormQuantOperator(NormQuantOperatorTestBase):
    def get_available_implementations(self, device):
        del device
        return []

    def _prepare_data_for_core_operator(self, *args, **kwargs):
        raise NotImplementedError

    def _execute_core_operator(self, *args, **kwargs):
        raise NotImplementedError


def _data(operator, *, tokens=2, hidden=4):
    return operator.generate_test_data(
        tokens=tokens,
        hidden=hidden,
        seed=7,
    )


def _patch_prepare(monkeypatch, operator, implementation, raw_op):
    monkeypatch.setattr(
        operator,
        "_resolve_implementation",
        lambda device, requested: implementation,
    )
    monkeypatch.setattr(operator, "_copy_to_device", _cpu_copy)
    if implementation.startswith("cuda_vllm"):
        monkeypatch.setattr(
            operator,
            "_resolve_vllm_callable",
            lambda symbol: raw_op,
        )
    else:
        monkeypatch.setattr(
            operator,
            "_resolve_flashinfer_callable",
            lambda symbol: raw_op,
        )


def _dynamic_vllm_oracle(x, residual, weight, eps):
    combined = x.float() + residual.float()
    rms = torch.rsqrt(combined.square().mean(dim=-1, keepdim=True) + eps)
    quant_values = ((combined * rms).to(x.dtype) * weight).float()
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    minimum_scale = 1.0 / (fp8_max * 512.0)
    scales = (
        quant_values.abs().amax(dim=-1, keepdim=True) / fp8_max
    ).clamp_min(minimum_scale)
    codes = (
        (quant_values / scales)
        .clamp(min=-fp8_max, max=fp8_max)
        .to(torch.float8_e4m3fn)
    )
    residual_out = combined.to(x.dtype)
    return codes, scales.to(torch.float32), residual_out


def _dynamic_fake(mode):
    def fake_raw(output, x, weight, scales, eps, bias, residual):
        assert bias is None
        codes, expected_scales, residual_out = _dynamic_vllm_oracle(
            x,
            residual,
            weight,
            eps,
        )
        if mode == "correct":
            output.copy_(codes)
            scales.copy_(expected_scales)
        elif mode == "same_dequant_wrong_scale":
            output.copy_((codes.float() / 2.0).to(output.dtype))
            scales.copy_(expected_scales * 2.0)
        elif mode == "saturated_to_416":
            tampered = torch.where(
                codes.float().abs() == 448.0,
                codes.float().sign() * 416.0,
                codes.float(),
            ).to(output.dtype)
            output.copy_(tampered)
            scales.copy_(expected_scales)
        elif mode == "zero_row_min_subnormal":
            output.copy_(codes)
            output[0, 0] = torch.tensor(2**-9, dtype=output.dtype)
            scales.copy_(expected_scales)
        else:
            raise AssertionError(f"unknown fake mode {mode}")
        residual.copy_(residual_out)

    return fake_raw


def test_common_generation_reference_and_byte_accounting_are_observable():
    operator = _operator(NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8)
    data = _data(operator, tokens=2, hidden=4)

    assert data["x"].shape == (2, 4)
    assert data["x"].dtype is torch.bfloat16
    assert data["residual"].shape == (2, 4)
    assert data["residual"].dtype is torch.bfloat16
    assert data["weight"].shape == (4,)
    assert data["weight"].dtype is torch.bfloat16
    assert data["eps"] == 1e-6
    assert data["metadata"] == {
        "tokens": 2,
        "hidden": 4,
        "variant": "add_rms_norm_dynamic_quant",
        "precision": "FP8",
    }

    combined = (
        data["x"].to(torch.bfloat16)
        + data["residual"].to(torch.bfloat16)
    ).float()
    expected = (
        combined
        * torch.rsqrt(
            combined.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        * data["weight"].float()
    )
    assert torch.equal(operator.run_cpu_reference(data), expected)

    # BF16 x/residual/weight reads, BF16 residual write, FP8 output write,
    # and one FP32 scale per token.
    assert operator.logical_bytes(data) == (
        2 * 4 * 2
        + 2 * 4 * 2
        + 4 * 2
        + 2 * 4 * 2
        + 2 * 4
        + 2 * 4
    )
    output = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    scales = torch.empty(2, 1, dtype=torch.float32)
    assert operator.physical_bytes(data, (output, scales)) == (
        2 * 4 * 2 + 2 * 4 * 2 + 4 * 2 + 2 * 4 * 2
        + output.numel() * output.element_size()
        + scales.numel() * scales.element_size()
    )
    assert operator.observable_output_bytes(
        (output, torch.empty(0), scales)
    ) == 16


def test_common_physical_bytes_count_bf16_x_out_once():
    operator = _CommonNormQuantOperator(
        NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
        PrecisionType.FP8,
    )
    data = _data(operator)
    quantized = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    x_out = torch.empty(2, 4, dtype=torch.bfloat16)

    assert operator.logical_bytes(data) == 68
    assert operator.physical_bytes(
        data,
        (quantized, torch.empty(0), x_out),
    ) == 68


def test_common_mxfp4_counts_pair_packing_and_e8m0_scale_bytes():
    operator = _CommonNormQuantOperator(
        NormQuantVariant.RMS_NORM_DYNAMIC_MX,
        PrecisionType.MXFP4,
    )
    data = _data(operator, tokens=2, hidden=64)
    packed = torch.empty(2, 32, dtype=torch.uint8)
    e8m0_scale = torch.empty(2, 1, 2, dtype=torch.uint8)

    assert operator.logical_bytes(data) == 452
    assert operator.physical_bytes(
        data,
        (packed, e8m0_scale, torch.empty(0)),
    ) == 452


@pytest.mark.parametrize(
    ("variant", "providers"),
    [
        (
            NormQuantVariant.RMS_NORM_STATIC_FP8,
            [VLLM_STATIC_RMS, FLASHINFER_STATIC_RMS],
        ),
        (
            NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
            [VLLM_STATIC_ADD, FLASHINFER_STATIC_ADD],
        ),
        (
            NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
            [VLLM_DYNAMIC_ADD],
        ),
    ],
)
def test_cuda_support_matrix_is_exact(monkeypatch, variant, providers):
    operator = _operator(variant)
    monkeypatch.setattr(
        operator,
        "_cuda_device_name",
        lambda device: "NVIDIA H20-3e",
    )

    assert operator.get_formal_implementations("cuda:4") == providers
    assert operator.supported_precisions == [PrecisionType.FP8]
    assert operator.operator_name == (
        f"NormQuant_{variant.value}_FP8"
    )


@pytest.mark.parametrize(
    "variant",
    [
        NormQuantVariant.RMS_NORM_DYNAMIC_MX,
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
    ],
)
def test_cuda_support_matrix_rejects_mx_variants(variant):
    with pytest.raises(ValueError, match="does not support"):
        create_norm_quant_operator("cuda:0", variant, PrecisionType.MXFP8)


def test_prepared_payloads_have_fresh_input_and_output_storage(monkeypatch):
    operator = _operator(NormQuantVariant.RMS_NORM_STATIC_FP8)
    _patch_prepare(
        monkeypatch,
        operator,
        VLLM_STATIC_RMS,
        lambda *args: None,
    )
    data = _data(operator)

    first = operator._prepare_data_for_core_operator(
        data, "cpu", PrecisionType.FP8, VLLM_STATIC_RMS
    )
    second = operator._prepare_data_for_core_operator(
        data, "cpu", PrecisionType.FP8, VLLM_STATIC_RMS
    )

    for key in ("x", "weight", "static_scale", "output"):
        assert first[key] is not second[key]
        assert (
            first[key].untyped_storage().data_ptr()
            != second[key].untyped_storage().data_ptr()
        )
    assert first["static_scale"].shape == (1,)
    assert first["static_scale"].dtype is torch.float32
    assert first["static_scale"].is_contiguous()


def test_vllm_static_rmsnorm_execute_uses_raw_out_first(monkeypatch):
    operator = _operator(NormQuantVariant.RMS_NORM_STATIC_FP8)
    observed = {}

    def fake_raw(output, x, weight, scale, eps):
        observed["args"] = (output, x, weight, scale, eps)
        output.copy_(torch.full_like(output, 3.0))

    _patch_prepare(monkeypatch, operator, VLLM_STATIC_RMS, fake_raw)
    prepared = operator._prepare_data_for_core_operator(
        _data(operator), "cpu", PrecisionType.FP8, VLLM_STATIC_RMS
    )

    result = operator._execute_core_operator(prepared, VLLM_STATIC_RMS)

    args = observed["args"]
    assert args[0] is prepared["output"]
    assert args[1] is prepared["x"]
    assert args[2] is prepared["weight"]
    assert args[3] is prepared["static_scale"]
    assert args[4] == prepared["eps"]
    assert result is prepared["output"]
    assert operator._declares_preallocated_output_contract(
        prepared, VLLM_STATIC_RMS
    )


def test_vllm_static_add_orders_residual_and_returns_only_output(monkeypatch):
    operator = _operator(NormQuantVariant.ADD_RMS_NORM_STATIC_FP8)
    observed = {}

    def fake_raw(output, x, residual, weight, scale, eps):
        observed["args"] = (output, x, residual, weight, scale, eps)
        output.copy_(torch.full_like(output, 2.0))
        residual.add_(x)

    _patch_prepare(monkeypatch, operator, VLLM_STATIC_ADD, fake_raw)
    prepared = operator._prepare_data_for_core_operator(
        _data(operator), "cpu", PrecisionType.FP8, VLLM_STATIC_ADD
    )

    result = operator._execute_core_operator(prepared, VLLM_STATIC_ADD)

    args = observed["args"]
    assert args[0] is prepared["output"]
    assert args[1] is prepared["x"]
    assert args[2] is prepared["residual"]
    assert args[3] is prepared["weight"]
    assert args[4] is prepared["static_scale"]
    assert args[5] == prepared["eps"]
    assert result is prepared["output"]
    assert prepared["residual"] is not prepared["output"]
    assert "residual" not in prepared.get("outputs", ())


def test_vllm_dynamic_add_uses_preallocated_fp8_and_fp32_scale_outputs(
    monkeypatch,
):
    operator = _operator(NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8)
    observed = {}

    def fake_raw(output, x, weight, scales, eps, bias, residual):
        observed["args"] = (
            output,
            x,
            weight,
            scales,
            eps,
            bias,
            residual,
        )
        output.copy_(torch.full_like(output, 1.0))
        scales.copy_(
            torch.arange(1, scales.shape[0] + 1, dtype=torch.float32)
            .reshape(-1, 1)
        )
        residual.add_(x)

    _patch_prepare(monkeypatch, operator, VLLM_DYNAMIC_ADD, fake_raw)
    prepared = operator._prepare_data_for_core_operator(
        _data(operator), "cpu", PrecisionType.FP8, VLLM_DYNAMIC_ADD
    )

    result = operator._execute_core_operator(prepared, VLLM_DYNAMIC_ADD)

    args = observed["args"]
    assert args[0] is prepared["outputs"][0]
    assert args[1] is prepared["x"]
    assert args[2] is prepared["weight"]
    assert args[3] is prepared["outputs"][1]
    assert args[4] == prepared["eps"]
    assert args[5] is None
    assert args[6] is prepared["residual"]
    assert result is prepared["outputs"]
    assert result[0].dtype is torch.float8_e4m3fn
    assert result[1].dtype is torch.float32
    assert result[1].shape == (2, 1)
    assert result[1].tolist() == [[1.0], [2.0]]


@pytest.mark.parametrize(
    ("variant", "implementation"),
    [
        (
            NormQuantVariant.RMS_NORM_STATIC_FP8,
            FLASHINFER_STATIC_RMS,
        ),
        (
            NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
            FLASHINFER_STATIC_ADD,
        ),
    ],
)
def test_flashinfer_static_passes_preallocated_scale_and_enable_pdl(
    monkeypatch,
    variant,
    implementation,
):
    operator = _operator(variant)
    observed = {}

    def fake_raw(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        args[0].copy_(torch.full_like(args[0], 4.0))
        if variant is NormQuantVariant.ADD_RMS_NORM_STATIC_FP8:
            args[2].add_(args[1])

    _patch_prepare(monkeypatch, operator, implementation, fake_raw)
    prepared = operator._prepare_data_for_core_operator(
        _data(operator), "cpu", PrecisionType.FP8, implementation
    )

    result = operator._execute_core_operator(prepared, implementation)

    assert observed["args"][0] is prepared["output"]
    assert observed["args"][-2] is prepared["static_scale"]
    assert observed["args"][-1] == prepared["eps"]
    assert observed["kwargs"] == {"enable_pdl": True}
    assert result is prepared["output"]
    assert operator._declares_preallocated_output_contract(
        prepared, implementation
    )


@pytest.mark.parametrize(
    ("variant", "implementation"),
    [
        (
            NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
            VLLM_STATIC_ADD,
        ),
        (
            NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
            VLLM_DYNAMIC_ADD,
        ),
    ],
)
def test_add_variants_restore_only_residual_from_seed(
    monkeypatch,
    variant,
    implementation,
):
    operator = _operator(variant)
    _patch_prepare(
        monkeypatch,
        operator,
        implementation,
        lambda *args: None,
    )
    prepared = operator._prepare_data_for_core_operator(
        _data(operator), "cpu", PrecisionType.FP8, implementation
    )
    x_after_capture = torch.full_like(prepared["x"], 9)
    prepared["x"].copy_(x_after_capture)
    prepared["residual"].fill_(11)
    expected_residual = prepared["residual_seed"].clone()

    operator._restore_mutable_graph_inputs(
        [prepared],
        implementation,
    )

    assert torch.equal(prepared["residual"], expected_residual)
    assert torch.equal(prepared["x"], x_after_capture)
    assert (
        prepared["residual"].untyped_storage().data_ptr()
        != prepared["residual_seed"].untyped_storage().data_ptr()
    )


@pytest.mark.parametrize(
    "device_name",
    [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H200",
        "NVIDIA H20",
        "NVIDIA H20-3e ",
        "nvidia h20-3e",
    ],
)
def test_h20_gate_rejects_other_sm90_device_names(
    monkeypatch,
    device_name,
):
    operator = _operator(NormQuantVariant.RMS_NORM_STATIC_FP8)
    monkeypatch.setattr(
        operator,
        "_cuda_device_name",
        lambda device: device_name,
    )

    assert operator.get_formal_implementations("cuda:0") == []
    with pytest.raises(ValueError, match="NVIDIA H20-3e"):
        operator._resolve_implementation("cuda:0", "default")


def test_hifp8_has_no_variant_or_provider(monkeypatch):
    assert "HIFP8" not in NormQuantVariant.__members__
    assert all("hifp8" not in variant.value for variant in NormQuantVariant)

    operator = _operator(NormQuantVariant.RMS_NORM_STATIC_FP8)
    monkeypatch.setattr(
        operator,
        "_cuda_device_name",
        lambda device: "NVIDIA H20-3e",
    )
    assert all(
        "hifp8" not in provider
        for provider in operator.get_formal_implementations("cuda:0")
    )


def test_auxiliary_correctness_rejects_bad_dynamic_scale_before_use(
    monkeypatch,
):
    operator = _operator(NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8)

    def fake_raw(output, x, weight, scales, eps, bias, residual):
        del weight, eps, bias
        output.zero_()
        scales.zero_()
        residual.add_(x)

    _patch_prepare(monkeypatch, operator, VLLM_DYNAMIC_ADD, fake_raw)
    data = _data(operator)
    prepared = operator._prepare_data_for_core_operator(
        data, "cpu", PrecisionType.FP8, VLLM_DYNAMIC_ADD
    )
    outputs = operator._execute_core_operator(prepared, VLLM_DYNAMIC_ADD)

    with pytest.raises(AssertionError, match="dynamic scale"):
        operator.validate_prepared_correctness(data, prepared, outputs)


def test_auxiliary_correctness_accepts_vllm_dynamic_scale_and_codes(
    monkeypatch,
):
    operator = _operator(NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8)
    _patch_prepare(
        monkeypatch,
        operator,
        VLLM_DYNAMIC_ADD,
        _dynamic_fake("correct"),
    )
    data = _data(operator)
    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.FP8,
        VLLM_DYNAMIC_ADD,
    )
    outputs = operator._execute_core_operator(prepared, VLLM_DYNAMIC_ADD)

    operator.validate_prepared_correctness(data, prepared, outputs)


def test_auxiliary_correctness_rejects_wrong_scale_with_same_dequant(
    monkeypatch,
):
    operator = _operator(NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8)
    _patch_prepare(
        monkeypatch,
        operator,
        VLLM_DYNAMIC_ADD,
        _dynamic_fake("same_dequant_wrong_scale"),
    )
    data = _data(operator)
    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.FP8,
        VLLM_DYNAMIC_ADD,
    )
    outputs = operator._execute_core_operator(prepared, VLLM_DYNAMIC_ADD)
    correct_codes, correct_scales, _ = _dynamic_vllm_oracle(
        prepared["x"],
        prepared["residual_seed"],
        prepared["weight"],
        prepared["eps"],
    )
    torch.testing.assert_close(
        outputs[0].float() * outputs[1],
        correct_codes.float() * correct_scales,
        rtol=0,
        atol=0,
    )

    with pytest.raises(AssertionError, match="dynamic scale semantics"):
        operator.validate_prepared_correctness(data, prepared, outputs)


def test_auxiliary_correctness_uses_vllm_zero_row_scale_semantics(
    monkeypatch,
):
    operator = _operator(NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8)
    _patch_prepare(
        monkeypatch,
        operator,
        VLLM_DYNAMIC_ADD,
        _dynamic_fake("correct"),
    )
    data = _data(operator)
    data["x"].zero_()
    data["residual"].zero_()
    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.FP8,
        VLLM_DYNAMIC_ADD,
    )
    outputs = operator._execute_core_operator(prepared, VLLM_DYNAMIC_ADD)

    operator.validate_prepared_correctness(data, prepared, outputs)
    torch.testing.assert_close(
        outputs[1],
        torch.full_like(outputs[1], 1.0 / (448.0 * 512.0)),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(outputs[0].float()) == 0


def test_auxiliary_correctness_rejects_unexpected_static_saturation(
    monkeypatch,
):
    operator = _operator(NormQuantVariant.RMS_NORM_STATIC_FP8)

    def fake_raw(output, x, weight, scale, eps):
        del x, weight, scale, eps
        output.fill_(torch.finfo(torch.float8_e4m3fn).max)

    _patch_prepare(monkeypatch, operator, VLLM_STATIC_RMS, fake_raw)
    data = _data(operator)
    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.FP8,
        VLLM_STATIC_RMS,
    )
    result = operator._execute_core_operator(prepared, VLLM_STATIC_RMS)

    with pytest.raises(AssertionError, match="static FP8 saturation"):
        operator.validate_prepared_correctness(data, prepared, result)


def test_auxiliary_correctness_rejects_adjacent_dynamic_saturation_code(
    monkeypatch,
):
    operator = _operator(NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8)
    _patch_prepare(
        monkeypatch,
        operator,
        VLLM_DYNAMIC_ADD,
        _dynamic_fake("saturated_to_416"),
    )
    data = _data(operator)
    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.FP8,
        VLLM_DYNAMIC_ADD,
    )
    outputs = operator._execute_core_operator(prepared, VLLM_DYNAMIC_ADD)
    assert bool((outputs[0].float().abs() == 416.0).any())
    assert not bool((outputs[0].float().abs() == 448.0).any())

    with pytest.raises(AssertionError, match="dynamic FP8 saturation"):
        operator.validate_prepared_correctness(data, prepared, outputs)


def test_auxiliary_correctness_rejects_nonzero_code_on_zero_row(
    monkeypatch,
):
    operator = _operator(NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8)
    _patch_prepare(
        monkeypatch,
        operator,
        VLLM_DYNAMIC_ADD,
        _dynamic_fake("zero_row_min_subnormal"),
    )
    data = _data(operator)
    data["x"].zero_()
    data["residual"].zero_()
    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.FP8,
        VLLM_DYNAMIC_ADD,
    )
    outputs = operator._execute_core_operator(prepared, VLLM_DYNAMIC_ADD)
    assert outputs[0][0, 0].float() == 2**-9
    torch.testing.assert_close(
        outputs[1],
        torch.full_like(outputs[1], 1.0 / (448.0 * 512.0)),
        rtol=0,
        atol=0,
    )

    with pytest.raises(AssertionError, match="dynamic FP8 zero-row"):
        operator.validate_prepared_correctness(data, prepared, outputs)


_NPU_SCHEMAS = {
    "npu_rms_norm_quant": (
        ("x", "gamma", "beta", "scale", "offset", "epsilon", "dst_dtype"),
        1,
    ),
    "npu_add_rms_norm_quant": (
        (
            "x1", "x2", "gamma", "scales1", "zero_points1", "beta",
            "scales2", "zero_points2", "axis", "epsilon", "div_mode",
            "dst_type",
        ),
        3,
    ),
    "npu_add_rms_norm_dynamic_quant": (
        (
            "x1", "x2", "gamma", "smooth_scale1", "smooth_scale2",
            "beta", "epsilon", "output_mask", "y_dtype",
        ),
        5,
    ),
    "npu_rms_norm_dynamic_mx_quant": (
        ("x", "gamma", "beta", "epsilon", "scale_alg", "round_mode",
         "dst_type"),
        3,
    ),
    "npu_add_rms_norm_dynamic_mx_quant": (
        ("x1", "x2", "gamma", "beta", "epsilon", "scale_alg",
         "round_mode", "dst_type"),
        4,
    ),
}


class _FakeNpuSchema:
    def __init__(self, name, arguments, return_arity):
        self.name = name
        self.arguments = tuple(
            SimpleNamespace(name=argument) for argument in arguments
        )
        self.returns = tuple(
            SimpleNamespace(name=f"output{index}")
            for index in range(return_arity)
        )

    def __str__(self):
        arguments = ", ".join(
            argument.name for argument in self.arguments
        )
        returns = ", ".join(output.name for output in self.returns)
        return f"npu::{self.name}({arguments}) -> ({returns})"


class _FakeNpuPacket:
    def __init__(self, schema):
        self._schemas = {"": schema}


def _fake_npu_ops(*, missing_fragment=None, return_arity=None):
    packets = {}
    for name, (arguments, arity) in _NPU_SCHEMAS.items():
        if missing_fragment is not None and name == missing_fragment[0]:
            arguments = tuple(
                argument
                for argument in arguments
                if argument != missing_fragment[1]
            )
        if return_arity is not None and name == return_arity[0]:
            arity = return_arity[1]
        packets[name] = _FakeNpuPacket(
            _FakeNpuSchema(name, arguments, arity)
        )
    return SimpleNamespace(**packets)


_FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _fake_rms_values(x, gamma, epsilon, residual=None):
    normalized_input = (
        x.float()
        if residual is None
        else (x.to(x.dtype) + residual.to(x.dtype)).float()
    )
    return (
        normalized_input
        * torch.rsqrt(
            normalized_input.square().mean(dim=-1, keepdim=True)
            + epsilon
        )
        * gamma.float()
    )


def _fake_dynamic_fp8(values):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (
        values.abs().amax(dim=-1) / fp8_max
    ).clamp_min(1.0 / (fp8_max * 512.0)).to(torch.float32)
    codes = (
        (values / scale.unsqueeze(-1))
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    return codes, scale


def _fake_mx_quant(values, dst_type):
    tokens, hidden = values.shape
    assert hidden % 64 == 0
    group_values = values.float().reshape(tokens, hidden // 32, 32)
    code_max = 448.0 if dst_type == 292 else 6.0
    raw_scale = group_values.abs().amax(dim=-1) / code_max
    exponent = torch.where(
        raw_scale > 0,
        torch.ceil(torch.log2(raw_scale)),
        torch.full_like(raw_scale, -127),
    ).clamp(-127, 128)
    scale_values = torch.pow(2.0, exponent)
    scale_bytes = (
        (exponent + 127)
        .to(torch.uint8)
        .reshape(tokens, hidden // 64, 2)
        .contiguous()
    )
    scaled = (group_values / scale_values.unsqueeze(-1)).reshape(
        tokens, hidden
    )
    if dst_type == 292:
        return scaled.clamp(-448, 448).to(torch.float8_e4m3fn), scale_bytes
    magnitudes = scaled.abs().clamp(max=6)
    indices = torch.argmin(
        (
            magnitudes.unsqueeze(-1)
            - _FP4_VALUES.to(magnitudes.device)
        ).abs(),
        dim=-1,
    ).to(torch.uint8)
    nibbles = indices | ((scaled < 0).to(torch.uint8) << 3)
    packed = (
        nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    ).contiguous()
    return packed, scale_bytes


class _FakeNpuRuntime:
    __version__ = "2.10.0.post1.dev20260528"
    float4_e2m1fn_x2 = 296
    float8_e8m0fnu = 293

    def __init__(self, *, failure=None):
        self.npu = SimpleNamespace(
            get_device_name=lambda index: "Ascend950PR_957b"
        )
        self.calls = []
        self.failure = failure

    def _record(self, name, *args, **kwargs):
        self.calls.append(
            {"name": name, "args": args, "kwargs": dict(kwargs)}
        )
        if self.failure is not None and self.failure[0] == name:
            raise self.failure[1]

    def npu_dynamic_quant(self, source, *, dst_type):
        self._record("npu_dynamic_quant", source, dst_type=dst_type)
        assert dst_type is torch.float8_e4m3fn
        return _fake_dynamic_fp8(source.float())

    def npu_dynamic_mx_quant(
        self,
        source,
        *,
        scale_alg,
        round_mode,
        dst_type,
    ):
        self._record(
            "npu_dynamic_mx_quant", source, scale_alg=scale_alg,
            round_mode=round_mode, dst_type=dst_type,
        )
        return _fake_mx_quant(source.float(), dst_type)

    def npu_rms_norm_quant(
        self,
        x,
        gamma,
        beta,
        scale,
        offset,
        epsilon=1e-6,
        *,
        dst_dtype=None,
    ):
        self._record(
            "npu_rms_norm_quant", x, gamma, beta, scale, offset, epsilon,
            dst_dtype=dst_dtype,
        )
        assert dst_dtype is torch.float8_e4m3fn
        values = _fake_rms_values(x, gamma, epsilon) + beta.float()
        return (values * scale.float() + offset.float()).to(dst_dtype)

    def npu_add_rms_norm_quant(
        self,
        x1,
        x2,
        gamma,
        scales1,
        zero_points1,
        beta=None,
        scales2=None,
        zero_points2=None,
        *,
        axis=-1,
        epsilon=1e-6,
        div_mode=True,
        dst_type=None,
    ):
        self._record(
            "npu_add_rms_norm_quant", x1, x2, gamma, scales1,
            zero_points1, beta, scales2, zero_points2, axis=axis,
            epsilon=epsilon, div_mode=div_mode, dst_type=dst_type,
        )
        assert dst_type == 292
        values = _fake_rms_values(
            x1, gamma, epsilon, residual=x2
        ) + beta.float()
        y1 = (values * scales1.float() + zero_points1.float()).to(
            torch.float8_e4m3fn
        )
        return (
            y1,
            torch.empty(0, dtype=torch.float8_e4m3fn),
            (x1.to(torch.bfloat16) + x2.to(torch.bfloat16)),
        )

    def npu_add_rms_norm_dynamic_quant(
        self,
        x1,
        x2,
        gamma,
        *,
        smooth_scale1=None,
        smooth_scale2=None,
        beta=None,
        epsilon=1e-6,
        output_mask=(),
        y_dtype=None,
    ):
        self._record(
            "npu_add_rms_norm_dynamic_quant", x1, x2, gamma,
            smooth_scale1=smooth_scale1, smooth_scale2=smooth_scale2,
            beta=beta, epsilon=epsilon, output_mask=list(output_mask),
            y_dtype=y_dtype,
        )
        values = _fake_rms_values(
            x1, gamma, epsilon, residual=x2
        ) + beta.float()
        y1, scale1 = _fake_dynamic_fp8(
            values.to(torch.bfloat16).float()
        )
        return (
            y1,
            torch.empty(0, dtype=torch.float8_e4m3fn),
            (x1.to(torch.bfloat16) + x2.to(torch.bfloat16)),
            scale1,
            torch.empty(0, dtype=torch.float32),
        )

    def npu_rms_norm_dynamic_mx_quant(
        self,
        x,
        gamma,
        *,
        beta=None,
        epsilon=1e-6,
        scale_alg=0,
        round_mode="rint",
        dst_type=296,
    ):
        self._record(
            "npu_rms_norm_dynamic_mx_quant", x, gamma, beta=beta,
            epsilon=epsilon, scale_alg=scale_alg, round_mode=round_mode,
            dst_type=dst_type,
        )
        values = _fake_rms_values(x, gamma, epsilon) + beta.float()
        quantized, scale = _fake_mx_quant(
            values.to(torch.bfloat16).float(),
            dst_type,
        )
        return quantized, scale, torch.empty(0, dtype=torch.float32)

    def npu_add_rms_norm_dynamic_mx_quant(
        self,
        x1,
        x2,
        gamma,
        *,
        beta=None,
        epsilon=1e-6,
        scale_alg=0,
        round_mode="rint",
        dst_type=296,
    ):
        self._record(
            "npu_add_rms_norm_dynamic_mx_quant", x1, x2, gamma,
            beta=beta, epsilon=epsilon, scale_alg=scale_alg,
            round_mode=round_mode, dst_type=dst_type,
        )
        values = _fake_rms_values(
            x1, gamma, epsilon, residual=x2
        ) + beta.float()
        quantized, scale = _fake_mx_quant(
            values.to(torch.bfloat16).float(),
            dst_type,
        )
        return (
            quantized,
            scale,
            (x1.to(torch.bfloat16) + x2.to(torch.bfloat16)),
            torch.empty(0, dtype=torch.float32),
        )


def _npu_operator(variant, precision):
    assert NpuNormQuantOperatorTest is not None, (
        "the Ascend 950PR NormQuant provider is missing"
    )
    return NpuNormQuantOperatorTest(variant, precision)


def _patch_npu_runtime(monkeypatch, operator, runtime, **ops_kwargs):
    monkeypatch.setattr(operator, "_load_torch_npu", lambda: runtime)
    monkeypatch.setattr(
        operator,
        "_copy_to_device",
        _cpu_copy,
    )
    monkeypatch.setattr(
        torch.ops,
        "npu",
        _fake_npu_ops(**ops_kwargs),
        raising=False,
    )


_NPU_PROVIDER_CASES = [
    (
        NormQuantVariant.RMS_NORM_STATIC_FP8,
        PrecisionType.FP8,
        "npu_rms_norm_quant_fp8_e4m3_static",
    ),
    (
        NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
        PrecisionType.FP8,
        "npu_add_rms_norm_quant_fp8_e4m3_static",
    ),
    (
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
        PrecisionType.FP8,
        "npu_add_rms_norm_dynamic_quant_fp8_e4m3_per_token",
    ),
    (
        NormQuantVariant.RMS_NORM_DYNAMIC_MX,
        PrecisionType.MXFP8,
        "npu_rms_norm_dynamic_mx_quant_mxfp8_e4m3_e8m0_g32",
    ),
    (
        NormQuantVariant.RMS_NORM_DYNAMIC_MX,
        PrecisionType.MXFP4,
        "npu_rms_norm_dynamic_mx_quant_mxfp4_e2m1_e8m0_g32",
    ),
    (
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
        PrecisionType.MXFP8,
        "npu_add_rms_norm_dynamic_mx_quant_mxfp8_e4m3_e8m0_g32",
    ),
    (
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
        PrecisionType.MXFP4,
        "npu_add_rms_norm_dynamic_mx_quant_mxfp4_e2m1_e8m0_g32",
    ),
]


@pytest.mark.parametrize(
    ("variant", "precision", "implementation"),
    _NPU_PROVIDER_CASES,
)
def test_npu_950pr_support_matrix_is_exact(
    monkeypatch,
    variant,
    precision,
    implementation,
):
    runtime = _FakeNpuRuntime()
    operator = _npu_operator(variant, precision)
    _patch_npu_runtime(monkeypatch, operator, runtime)

    assert operator.get_formal_implementations("npu:3") == [
        implementation
    ]
    assert operator.get_available_implementations("npu:3") == [
        implementation
    ]
    assert operator.supported_precisions == [precision]
    assert isinstance(
        create_norm_quant_operator("npu:3", variant, precision),
        NpuNormQuantOperatorTest,
    )


@pytest.mark.parametrize(
    "device_name",
    [
        "Ascend910C",
        "ascend950PR_957b",
        "EmulatedAscend950PR_957b",
        " Ascend950PR_957b",
    ],
)
def test_npu_950pr_gate_rejects_nonexact_prefix(
    monkeypatch,
    device_name,
):
    operator = _npu_operator(
        NormQuantVariant.RMS_NORM_STATIC_FP8,
        PrecisionType.FP8,
    )
    runtime = _FakeNpuRuntime()
    _patch_npu_runtime(monkeypatch, operator, runtime)
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: device_name,
    )

    assert operator.get_formal_implementations("npu:0") == []


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("missing_symbol", "npu_add_rms_norm_dynamic_quant"),
        (
            "missing_fragment",
            ("npu_add_rms_norm_dynamic_quant", "y_dtype"),
        ),
        (
            "return_arity",
            ("npu_add_rms_norm_dynamic_quant", 4),
        ),
    ],
)
def test_npu_formal_gate_rejects_missing_symbol_schema_or_return_arity(
    monkeypatch,
    mutation,
    value,
):
    operator = _npu_operator(
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
        PrecisionType.FP8,
    )
    runtime = _FakeNpuRuntime()
    if mutation == "missing_symbol":
        setattr(runtime, value, None)
        ops_kwargs = {}
    else:
        ops_kwargs = {mutation: value}
    _patch_npu_runtime(monkeypatch, operator, runtime, **ops_kwargs)

    assert operator.get_formal_implementations("npu:0") == []


def test_npu_formal_gate_rejects_missing_e4m3_dtype(monkeypatch):
    operator = _npu_operator(
        NormQuantVariant.RMS_NORM_STATIC_FP8,
        PrecisionType.FP8,
    )
    runtime = _FakeNpuRuntime()
    _patch_npu_runtime(monkeypatch, operator, runtime)
    monkeypatch.setattr(
        operator,
        "_fp8_dtype",
        lambda: None,
        raising=False,
    )

    assert operator.get_formal_implementations("npu:0") == []


def test_npu_capability_probe_is_instance_cached_by_runtime_key(
    monkeypatch,
):
    operator = _npu_operator(
        NormQuantVariant.RMS_NORM_DYNAMIC_MX,
        PrecisionType.MXFP8,
    )
    runtime = _FakeNpuRuntime()
    _patch_npu_runtime(monkeypatch, operator, runtime)

    assert operator.get_formal_implementations("npu:1")
    calls_after_first = len(runtime.calls)
    assert operator.get_formal_implementations("npu:1")
    assert len(runtime.calls) == calls_after_first
    assert list(operator._capability_cache) == [
        (
            "npu:1",
            runtime.__version__,
            "npu_rms_norm_dynamic_mx_quant",
            292,
        )
    ]


def test_npu_probe_failure_preserves_original_error_without_int8_fallback(
    monkeypatch,
):
    probe_error = RuntimeError("native E4M3 is not implemented")
    runtime = _FakeNpuRuntime(
        failure=("npu_add_rms_norm_quant", probe_error)
    )
    operator = _npu_operator(
        NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
        PrecisionType.FP8,
    )
    _patch_npu_runtime(monkeypatch, operator, runtime)

    assert operator.get_formal_implementations("npu:0") == []
    assert operator.capability_error is probe_error
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["kwargs"]["dst_type"] == 292
    assert all(
        call["kwargs"].get("dst_type") not in (torch.int8, torch.qint8)
        for call in runtime.calls
    )


def test_npu_probe_rejects_int8_result_for_requested_e4m3(monkeypatch):
    runtime = _FakeNpuRuntime()

    def wrong_dtype(
        x,
        gamma,
        beta,
        scale,
        offset,
        epsilon=1e-6,
        *,
        dst_dtype=None,
    ):
        del gamma, beta, scale, offset, epsilon, dst_dtype
        return torch.zeros_like(x, dtype=torch.int8)

    runtime.npu_rms_norm_quant = wrong_dtype
    operator = _npu_operator(
        NormQuantVariant.RMS_NORM_STATIC_FP8,
        PrecisionType.FP8,
    )
    _patch_npu_runtime(monkeypatch, operator, runtime)

    assert operator.get_formal_implementations("npu:0") == []
    assert isinstance(operator.capability_error, RuntimeError)
    assert "dtype" in str(operator.capability_error)


@pytest.mark.parametrize(
    ("variant", "precision", "implementation"),
    _NPU_PROVIDER_CASES,
)
def test_npu_native_calls_retain_full_allocated_output_contract(
    monkeypatch,
    variant,
    precision,
    implementation,
):
    runtime = _FakeNpuRuntime()
    operator = _npu_operator(variant, precision)
    _patch_npu_runtime(monkeypatch, operator, runtime)
    assert operator.get_formal_implementations("npu:0") == [implementation]
    runtime.calls.clear()
    prepared = operator._prepare_data_for_core_operator(
        _data(operator, tokens=2, hidden=64),
        "npu:0",
        precision,
        implementation,
    )

    result = operator._execute_core_operator(prepared, implementation)
    outputs = result if isinstance(result, tuple) else (result,)

    expected_arity = {
        NormQuantVariant.RMS_NORM_STATIC_FP8: 1,
        NormQuantVariant.ADD_RMS_NORM_STATIC_FP8: 3,
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8: 5,
        NormQuantVariant.RMS_NORM_DYNAMIC_MX: 3,
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX: 4,
    }[variant]
    assert len(outputs) == expected_arity
    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert "outputs" not in prepared
    assert "output" not in prepared
    assert not operator._declares_preallocated_output_contract(
        prepared,
        implementation,
    )
    call = runtime.calls[-1]
    assert "out" not in call["kwargs"]


@pytest.mark.parametrize(
    ("variant", "precision", "implementation"),
    _NPU_PROVIDER_CASES,
)
def test_npu_native_calls_follow_950pr_abi_snapshot(
    monkeypatch,
    variant,
    precision,
    implementation,
):
    runtime = _FakeNpuRuntime()
    operator = _npu_operator(variant, precision)
    _patch_npu_runtime(monkeypatch, operator, runtime)
    assert operator.get_formal_implementations("npu:0") == [implementation]
    runtime.calls.clear()
    prepared = operator._prepare_data_for_core_operator(
        _data(operator, tokens=2, hidden=64),
        "npu:0",
        precision,
        implementation,
    )

    operator._execute_core_operator(prepared, implementation)

    call = runtime.calls[-1]
    native_op = {
        NormQuantVariant.RMS_NORM_STATIC_FP8: "npu_rms_norm_quant",
        NormQuantVariant.ADD_RMS_NORM_STATIC_FP8:
        "npu_add_rms_norm_quant",
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8:
        "npu_add_rms_norm_dynamic_quant",
        NormQuantVariant.RMS_NORM_DYNAMIC_MX:
        "npu_rms_norm_dynamic_mx_quant",
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX:
        "npu_add_rms_norm_dynamic_mx_quant",
    }[variant]
    assert call["name"] == native_op
    assert len(call["args"]) == {
        NormQuantVariant.RMS_NORM_STATIC_FP8: 6,
        NormQuantVariant.ADD_RMS_NORM_STATIC_FP8: 8,
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8: 3,
        NormQuantVariant.RMS_NORM_DYNAMIC_MX: 2,
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX: 3,
    }[variant]
    if variant is NormQuantVariant.RMS_NORM_STATIC_FP8:
        expected_kwargs = {"dst_dtype": torch.float8_e4m3fn}
    elif variant is NormQuantVariant.ADD_RMS_NORM_STATIC_FP8:
        expected_kwargs = {
            "axis": -1, "epsilon": 1e-6, "div_mode": True,
            "dst_type": 292,
        }
    elif variant is NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8:
        expected_kwargs = {
            "smooth_scale1": None, "smooth_scale2": None,
            "epsilon": 1e-6, "output_mask": [True, False],
            "y_dtype": torch.float8_e4m3fn,
        }
    else:
        expected_kwargs = {
            "epsilon": 1e-6, "scale_alg": 0, "round_mode": "rint",
            "dst_type": 296 if precision is PrecisionType.MXFP4 else 292,
        }
    assert call["kwargs"] | expected_kwargs == call["kwargs"]
    for name, value in expected_kwargs.items():
        assert call["kwargs"][name] is value or call["kwargs"][name] == value
    if "beta" in call["kwargs"]:
        assert call["kwargs"]["beta"] is prepared["beta"]
    elif variant is not NormQuantVariant.RMS_NORM_STATIC_FP8:
        assert call["args"][5] is prepared["beta"]


@pytest.mark.parametrize(
    "precision",
    [PrecisionType.MXFP8, PrecisionType.MXFP4],
)
def test_npu_mx_requires_hidden_multiple_of_64(
    monkeypatch,
    precision,
):
    operator = _npu_operator(
        NormQuantVariant.RMS_NORM_DYNAMIC_MX,
        precision,
    )
    runtime = _FakeNpuRuntime()
    _patch_npu_runtime(monkeypatch, operator, runtime)

    with pytest.raises(ValueError, match="multiple of 64"):
        operator._prepare_data_for_core_operator(
            _data(operator, tokens=2, hidden=96),
            "npu:0",
            precision,
        )


def _run_fake_npu_provider(monkeypatch, variant, precision):
    runtime = _FakeNpuRuntime()
    operator = _npu_operator(variant, precision)
    _patch_npu_runtime(monkeypatch, operator, runtime)
    data = _data(operator, tokens=2, hidden=64)
    prepared = operator._prepare_data_for_core_operator(
        data,
        "npu:0",
        precision,
    )
    result = operator._execute_core_operator(prepared)
    return operator, runtime, data, prepared, result


@pytest.mark.parametrize(
    ("variant", "precision"),
    [(variant, precision)
     for variant, precision, _ in _NPU_PROVIDER_CASES],
)
def test_npu_auxiliary_correctness_accepts_real_output_tensors(
    monkeypatch,
    variant,
    precision,
):
    operator, _, data, prepared, result = _run_fake_npu_provider(
        monkeypatch,
        variant,
        precision,
    )

    operator.validate_prepared_correctness(data, prepared, result)


def test_npu_correctness_rejects_bad_plain_primary_dequant(monkeypatch):
    operator, _, data, prepared, result = _run_fake_npu_provider(
        monkeypatch,
        NormQuantVariant.RMS_NORM_STATIC_FP8,
        PrecisionType.FP8,
    )
    result.zero_()

    with pytest.raises(AssertionError):
        operator.validate_prepared_correctness(data, prepared, result)


def test_npu_correctness_rejects_bad_dynamic_plain_scale(monkeypatch):
    operator, _, data, prepared, result = _run_fake_npu_provider(
        monkeypatch,
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
        PrecisionType.FP8,
    )
    result[3].mul_(2)

    with pytest.raises(AssertionError, match="dynamic FP8 scale"):
        operator.validate_prepared_correctness(data, prepared, result)


def test_npu_correctness_rejects_bad_add_x_out(monkeypatch):
    operator, _, data, prepared, result = _run_fake_npu_provider(
        monkeypatch,
        NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
        PrecisionType.FP8,
    )
    result[2].zero_()

    with pytest.raises(AssertionError, match="x_out"):
        operator.validate_prepared_correctness(data, prepared, result)


def test_npu_correctness_rejects_bad_mxfp8_e8m0_scale(monkeypatch):
    operator, _, data, prepared, result = _run_fake_npu_provider(
        monkeypatch,
        NormQuantVariant.RMS_NORM_DYNAMIC_MX,
        PrecisionType.MXFP8,
    )
    result[1].add_(1)

    with pytest.raises(AssertionError, match="E8M0 scale"):
        operator.validate_prepared_correctness(data, prepared, result)


def test_npu_correctness_checks_both_mxfp4_values_per_packed_byte(
    monkeypatch,
):
    operator, _, data, prepared, result = _run_fake_npu_provider(
        monkeypatch,
        NormQuantVariant.RMS_NORM_DYNAMIC_MX,
        PrecisionType.MXFP4,
    )
    result[0][0, 0] ^= 0x10

    with pytest.raises(AssertionError, match="packed E2M1"):
        operator.validate_prepared_correctness(data, prepared, result)


def test_npu_correctness_rejects_nonempty_disabled_output(monkeypatch):
    operator, _, data, prepared, result = _run_fake_npu_provider(
        monkeypatch,
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
        PrecisionType.FP8,
    )
    bad_result = (
        result[0],
        torch.ones(1, dtype=torch.float8_e4m3fn),
        result[2],
        result[3],
        result[4],
    )

    with pytest.raises(AssertionError, match="disabled"):
        operator.validate_prepared_correctness(
            data,
            prepared,
            bad_result,
        )
