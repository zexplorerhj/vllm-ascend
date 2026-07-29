from pathlib import Path
import sys

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import PrecisionType  # noqa: E402
from groupgemm.groupgemm_fp8_npu import (  # noqa: E402
    GroupGemmFp8NpuOperatorTest,
)
from groupgemm.groupgemm_mxfp8_npu import (  # noqa: E402
    GroupGemmMxFp8NpuOperatorTest,
)
from groupgemm.groupgemm_mxfp4_npu import (  # noqa: E402
    GroupGemmMxFp4NpuOperatorTest,
)
import groupgemm.groupgemm_fp8_npu as fp8_npu_module  # noqa: E402
import groupgemm.groupgemm_mxfp8_npu as mxfp8_npu_module  # noqa: E402
import groupgemm.groupgemm_mxfp4_npu as mxfp4_npu_module  # noqa: E402


class FakeNpuRuntime:
    float4_e2m1fn_x2 = 296
    float8_e8m0fnu = 293

    def __init__(self):
        self.dynamic_quant_inputs = []
        self.dynamic_mx_quant_inputs = []
        self.dynamic_mx_quant_calls = []
        self.dynamic_mx_quant_outputs = []
        self.grouped_matmul_calls = []

    def npu_dynamic_quant(self, source, *, dst_type):
        self.dynamic_quant_inputs.append(source)
        scale = torch.ones(source.shape[:-1], dtype=torch.float32)
        return source.to(dst_type), scale

    def npu_dynamic_mx_quant(
        self,
        source,
        *,
        dst_type,
        block_size=None,
        round_mode=None,
    ):
        self.dynamic_mx_quant_inputs.append(source)
        self.dynamic_mx_quant_calls.append(
            {
                "source": source,
                "dst_type": dst_type,
                "block_size": block_size,
                "round_mode": round_mode,
            }
        )
        scale_shape = (*source.shape[:-1], source.shape[-1] // 64, 2)
        scale = torch.empty(scale_shape, dtype=torch.uint8)
        scale.flatten().copy_(
            torch.arange(scale.numel(), dtype=torch.int64)
            .remainder(251)
            .to(torch.uint8)
        )
        if dst_type == self.float4_e2m1fn_x2:
            quantized = torch.empty(
                *source.shape[:-1],
                source.shape[-1] // 2,
                dtype=torch.uint8,
            )
            quantized.flatten().copy_(
                torch.arange(quantized.numel(), dtype=torch.int64)
                .remainder(251)
                .to(torch.uint8)
            )
        else:
            quantized = source.to(dst_type)
        self.dynamic_mx_quant_outputs.append((quantized, scale))
        return quantized, scale

    def npu_grouped_matmul(self, **kwargs):
        self.grouped_matmul_calls.append(kwargs)
        rows = kwargs["x"][0].shape[0]
        out_channel = kwargs["weight"][0].shape[-1]
        return [
            torch.empty(
                rows,
                out_channel,
                dtype=kwargs["output_dtype"],
            )
        ]


def _cpu_copy(tensor, *, device, dtype=None):
    del device
    return tensor.to(dtype=dtype or tensor.dtype).clone()


def _patch_runtime(monkeypatch, module, operator, runtime):
    monkeypatch.setattr(module, "torch_npu", runtime)
    monkeypatch.setattr(operator, "_copy_to_device", _cpu_copy)
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "Ascend950PR_957b",
    )


def test_plain_fp8_provider_uses_one_full_weight_quant_and_pure_gmm2(
    monkeypatch,
):
    runtime = FakeNpuRuntime()
    operator = GroupGemmFp8NpuOperatorTest(
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )
    _patch_runtime(monkeypatch, fp8_npu_module, operator, runtime)
    data = operator.generate_test_data(
        seq_len=4,
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )

    prepared = operator._prepare_data_for_core_operator(
        data,
        "npu:0",
        PrecisionType.FP8,
        operator.NPU_FP8_IMPLEMENTATION,
    )
    output = operator._execute_core_operator(
        prepared,
        operator.NPU_FP8_IMPLEMENTATION,
    )

    assert [tuple(value.shape) for value in runtime.dynamic_quant_inputs] == [
        (4, 64),
        (2, 32, 64),
    ]
    assert prepared["x"].shape == (4, 64)
    assert prepared["weight"].shape == (2, 64, 32)
    assert prepared["per_token_scale"].shape == (4,)
    assert prepared["weight_scale"].shape == (2, 32)
    assert prepared["x"].dtype == torch.float8_e4m3fn
    assert prepared["weight"].dtype == torch.float8_e4m3fn
    assert prepared["per_token_scale"].dtype == torch.float32
    assert prepared["weight_scale"].dtype == torch.float32
    assert output.shape == (4, 32)
    assert output.dtype == torch.bfloat16

    call = runtime.grouped_matmul_calls[0]
    assert call["x"] == [prepared["x"]]
    assert call["weight"] == [prepared["weight"]]
    assert call["scale"] == [prepared["weight_scale"]]
    assert call["per_token_scale"] == [prepared["per_token_scale"]]
    assert call["group_list"] is prepared["group_list"]
    assert call["split_item"] == 2
    assert call["group_list_type"] == 1
    assert call["group_type"] == 0
    assert call["output_dtype"] == torch.bfloat16
    assert call["bias"] is None
    assert "x_dtype" not in call
    assert "weight_dtype" not in call
    assert "scale_dtype" not in call
    assert "per_token_scale_dtype" not in call


def test_mxfp8_provider_uses_e8m0_group32_layout_and_pure_gmm2(
    monkeypatch,
):
    runtime = FakeNpuRuntime()
    operator = GroupGemmMxFp8NpuOperatorTest(
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )
    _patch_runtime(monkeypatch, mxfp8_npu_module, operator, runtime)
    data = operator.generate_test_data(
        seq_len=4,
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )

    prepared = operator._prepare_data_for_core_operator(
        data,
        "npu:0",
        PrecisionType.MXFP8,
        operator.NPU_MXFP8_IMPLEMENTATION,
    )
    output = operator._execute_core_operator(
        prepared,
        operator.NPU_MXFP8_IMPLEMENTATION,
    )

    assert [
        tuple(value.shape) for value in runtime.dynamic_mx_quant_inputs
    ] == [
        (4, 64),
        (2, 32, 64),
    ]
    assert prepared["x"].shape == (4, 64)
    assert prepared["weight"].shape == (2, 64, 32)
    assert prepared["per_token_scale"].shape == (4, 1, 2)
    assert prepared["weight_scale"].shape == (2, 1, 32, 2)
    assert prepared["per_token_scale"].dtype == torch.uint8
    assert prepared["weight_scale"].dtype == torch.uint8
    assert output.shape == (4, 32)
    assert output.dtype == torch.bfloat16

    expected_scale = torch.arange(
        2 * 32 * 1 * 2,
        dtype=torch.uint8,
    ).reshape(2, 32, 1, 2).transpose(1, 2).contiguous()
    assert torch.equal(prepared["weight_scale"], expected_scale)

    call = runtime.grouped_matmul_calls[0]
    assert call["x"] == [prepared["x"]]
    assert call["weight"] == [prepared["weight"]]
    assert call["scale"] == [prepared["weight_scale"]]
    assert call["per_token_scale"] == [prepared["per_token_scale"]]
    assert call["group_list"] is prepared["group_list"]
    assert call["split_item"] == 2
    assert call["group_list_type"] == 1
    assert call["group_type"] == 0
    assert call["output_dtype"] == torch.bfloat16
    assert call["scale_dtype"] == runtime.float8_e8m0fnu
    assert call["per_token_scale_dtype"] == runtime.float8_e8m0fnu
    assert "x_dtype" not in call
    assert "weight_dtype" not in call
    assert not hasattr(runtime, "npu_grouped_matmul_swiglu_quant_v2")


def test_mxfp4_provider_preserves_native_packed_views_and_calls_one_gmm(
    monkeypatch,
):
    runtime = FakeNpuRuntime()
    operator = GroupGemmMxFp4NpuOperatorTest(
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )
    _patch_runtime(monkeypatch, mxfp4_npu_module, operator, runtime)
    data = operator.generate_test_data(
        seq_len=4,
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )

    prepared = operator._prepare_data_for_core_operator(
        data,
        "npu:0",
        PrecisionType.MXFP4,
        operator.NPU_MXFP4_IMPLEMENTATION,
    )
    output = operator._execute_core_operator(
        prepared,
        operator.NPU_MXFP4_IMPLEMENTATION,
    )

    assert [
        (
            tuple(call["source"].shape),
            call["source"].dtype,
            call["source"].is_contiguous(),
            call["dst_type"],
            call["block_size"],
            call["round_mode"],
        )
        for call in runtime.dynamic_mx_quant_calls
    ] == [
        ((4, 64), torch.bfloat16, True, 296, 32, "round"),
        ((2, 32, 64), torch.bfloat16, True, 296, 32, "round"),
    ]
    assert prepared["x"].shape == (4, 32)
    assert prepared["x"].dtype is torch.uint8
    assert prepared["x"].stride() == (32, 1)
    assert prepared["per_token_scale"].shape == (4, 1, 2)
    assert prepared["per_token_scale"].dtype is torch.uint8

    weight_packed_enk, weight_scale_enk = (
        runtime.dynamic_mx_quant_outputs[1]
    )
    assert prepared["weight"].shape == (2, 32, 32)
    assert prepared["weight"].dtype is torch.uint8
    assert prepared["weight"].stride() == (1024, 1, 32)
    assert prepared["weight"]._base is weight_packed_enk
    assert prepared["weight"].untyped_storage().data_ptr() == (
        weight_packed_enk.untyped_storage().data_ptr()
    )
    assert prepared["weight_scale"].shape == (2, 1, 32, 2)
    assert prepared["weight_scale"].dtype is torch.uint8
    assert prepared["weight_scale"].stride() == (64, 2, 2, 1)
    assert prepared["weight_scale"]._base is weight_scale_enk
    assert prepared["weight_scale"].untyped_storage().data_ptr() == (
        weight_scale_enk.untyped_storage().data_ptr()
    )
    assert prepared["group_list"].tolist() == [2, 2]
    assert output.shape == (4, 32)
    assert output.dtype is torch.bfloat16

    assert len(runtime.grouped_matmul_calls) == 1
    call = runtime.grouped_matmul_calls[0]
    assert call == {
        "x": [prepared["x"]],
        "weight": [prepared["weight"]],
        "scale": [prepared["weight_scale"]],
        "bias": None,
        "per_token_scale": [prepared["per_token_scale"]],
        "split_item": 2,
        "group_list_type": 1,
        "group_type": 0,
        "group_list": prepared["group_list"],
        "output_dtype": torch.bfloat16,
        "x_dtype": 296,
        "weight_dtype": 296,
        "scale_dtype": 293,
        "per_token_scale_dtype": 293,
    }
    assert len(call["x"]) == len(call["weight"]) == 1
    assert "group_sizes" not in call
    assert not hasattr(runtime, "npu_grouped_matmul_swiglu_quant_v2")


@pytest.mark.parametrize(
    ("provider_class", "module", "implementation"),
    [
        (
            GroupGemmFp8NpuOperatorTest,
            fp8_npu_module,
            (
                "npu_grouped_matmul_fp8_e4m3_"
                "per_token_per_channel_bf16"
            ),
        ),
        (
            GroupGemmMxFp8NpuOperatorTest,
            mxfp8_npu_module,
            (
                "npu_grouped_matmul_mxfp8_e4m3_"
                "e8m0_group32_bf16"
            ),
        ),
        (
            GroupGemmMxFp4NpuOperatorTest,
            mxfp4_npu_module,
            (
                "npu_grouped_matmul_mxfp4_e2m1_"
                "e8m0_group32_bf16"
            ),
        ),
    ],
)
def test_npu_fp8_providers_are_formal_only_on_950pr(
    monkeypatch,
    provider_class,
    module,
    implementation,
):
    runtime = FakeNpuRuntime()
    operator = provider_class(
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )
    monkeypatch.setattr(module, "torch_npu", runtime)
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "Ascend950PR_957b",
    )

    assert operator.get_formal_implementations("npu:0") == [implementation]
    assert operator.get_available_implementations("npu:0") == [
        implementation
    ]
    assert operator.get_formal_implementations("cuda:0") == []

    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "Ascend910C",
    )
    assert operator.get_formal_implementations("npu:0") == []


def test_npu_fp8_provider_rejects_nonprefix_950pr_device_name(monkeypatch):
    runtime = FakeNpuRuntime()
    operator = GroupGemmFp8NpuOperatorTest()
    monkeypatch.setattr(fp8_npu_module, "torch_npu", runtime)
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "EmulatedAscend950PR_957b",
    )

    assert operator.get_formal_implementations("npu:0") == []


@pytest.mark.parametrize(
    "runtime",
    [
        type(
            "MissingDynamicMxQuant",
            (),
            {
                "float4_e2m1fn_x2": 296,
                "float8_e8m0fnu": 293,
                "npu_grouped_matmul": lambda *args, **kwargs: None,
            },
        )(),
        type(
            "MissingGroupedMatmul",
            (),
            {
                "float4_e2m1fn_x2": 296,
                "float8_e8m0fnu": 293,
                "npu_dynamic_mx_quant": lambda *args, **kwargs: None,
            },
        )(),
        type(
            "MissingMxFp4Dtype",
            (),
            {
                "float8_e8m0fnu": 293,
                "npu_dynamic_mx_quant": lambda *args, **kwargs: None,
                "npu_grouped_matmul": lambda *args, **kwargs: None,
            },
        )(),
        type(
            "MissingE8M0Dtype",
            (),
            {
                "float4_e2m1fn_x2": 296,
                "npu_dynamic_mx_quant": lambda *args, **kwargs: None,
                "npu_grouped_matmul": lambda *args, **kwargs: None,
            },
        )(),
    ],
)
def test_mxfp4_provider_gate_requires_native_symbols_and_dtype_codes(
    monkeypatch,
    runtime,
):
    operator = GroupGemmMxFp4NpuOperatorTest()
    monkeypatch.setattr(mxfp4_npu_module, "torch_npu", runtime)
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "Ascend950PR_957b",
    )

    assert operator.get_formal_implementations("npu:0") == []


def test_fp8_mxfp8_and_mxfp4_advertise_distinct_precision_tokens():
    plain = GroupGemmFp8NpuOperatorTest()
    mxfp8 = GroupGemmMxFp8NpuOperatorTest()
    mxfp4 = GroupGemmMxFp4NpuOperatorTest()

    assert plain.supported_precisions == [PrecisionType.FP8]
    assert mxfp8.supported_precisions == [PrecisionType.MXFP8]
    assert mxfp4.supported_precisions == [PrecisionType.MXFP4]


def test_quantized_groupgemm_bandwidth_accounts_for_packed_mxfp4_data():
    plain = GroupGemmFp8NpuOperatorTest(
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )
    mx = GroupGemmMxFp8NpuOperatorTest(
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )
    mxfp4 = GroupGemmMxFp4NpuOperatorTest(
        num_experts=2,
        hidden_dim=64,
        out_channel=32,
    )
    data = {
        "seq_len": 4,
        "num_experts": 2,
        "hidden_dim": 64,
        "out_channel": 32,
    }

    assert plain.calculate_bandwidth(data, 1.0) == pytest.approx(
        4880 / 1e6
    )
    assert mx.calculate_bandwidth(data, 1.0) == pytest.approx(
        4744 / 1e6
    )
    assert mxfp4.calculate_bandwidth(data, 1.0) == pytest.approx(
        2568 / 1e6
    )
