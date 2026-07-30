from pathlib import Path
import sys

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
