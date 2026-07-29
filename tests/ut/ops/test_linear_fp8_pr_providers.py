from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

try:
    from linear.linear_fp8_npu_operator import LinearFp8NpuOperatorTest
except ModuleNotFoundError:
    LinearFp8NpuOperatorTest = None

try:
    from linear.linear_mxfp8_npu_operator import LinearMxFp8NpuOperatorTest
except ModuleNotFoundError:
    LinearMxFp8NpuOperatorTest = None

try:
    from linear.linear_mxfp4_npu_operator import LinearMxFp4NpuOperatorTest
except ModuleNotFoundError:
    LinearMxFp4NpuOperatorTest = None

from operator_test_framework import PrecisionType  # noqa: E402


PLAIN_IMPLEMENTATION = (
    "npu_quant_matmul_fp8_e4m3_per_token_per_channel_bf16"
)
MXFP8_IMPLEMENTATION = (
    "npu_quant_matmul_mxfp8_e4m3_e8m0_group32_bf16"
)
MXFP4_IMPLEMENTATION = (
    "npu_quant_matmul_mxfp4_e2m1_e8m0_group32_bf16"
)


def _plain_operator():
    assert LinearFp8NpuOperatorTest is not None, (
        "the independent Ascend 950PR plain FP8 Linear provider is missing"
    )
    return LinearFp8NpuOperatorTest()


def _mxfp8_operator():
    assert LinearMxFp8NpuOperatorTest is not None, (
        "the independent Ascend 950PR MXFP8 Linear provider is missing"
    )
    return LinearMxFp8NpuOperatorTest()


def _mxfp4_operator():
    assert LinearMxFp4NpuOperatorTest is not None, (
        "the independent Ascend 950PR MXFP4 Linear provider is missing"
    )
    return LinearMxFp4NpuOperatorTest()


def test_950pr_linear_providers_are_independent_public_classes():
    plain = _plain_operator()
    mxfp8 = _mxfp8_operator()
    mxfp4 = _mxfp4_operator()

    assert type(plain) is not type(mxfp8)
    assert type(mxfp4) not in {type(plain), type(mxfp8)}
    assert plain.NPU_IMPLEMENTATION == PLAIN_IMPLEMENTATION
    assert mxfp8.NPU_IMPLEMENTATION == MXFP8_IMPLEMENTATION
    assert mxfp4.NPU_IMPLEMENTATION == MXFP4_IMPLEMENTATION
    assert plain.operator_name == "LinearFp8Npu"
    assert mxfp8.operator_name == "LinearMxFp8Npu"
    assert mxfp4.operator_name == "LinearMxFp4Npu"
    assert plain.supported_precisions == [PrecisionType.FP8]
    assert mxfp8.supported_precisions == [PrecisionType.MXFP8]
    assert mxfp4.supported_precisions == [PrecisionType.MXFP4]


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("Ascend950PR_957b", True),
        ("Ascend950PR", True),
        ("Ascend950DT", False),
        ("Ascend910C", False),
    ],
)
def test_plain_fp8_provider_is_formal_only_on_ascend_950pr(
    monkeypatch,
    device_name,
    expected,
):
    operator = _plain_operator()
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: device_name,
    )
    monkeypatch.setattr(
        operator,
        "_runtime_supported",
        lambda: True,
        raising=False,
    )

    assert operator.get_formal_implementations("npu:3") == (
        [PLAIN_IMPLEMENTATION] if expected else []
    )
    if expected:
        assert operator._resolve_implementation("npu:3", "default") == (
            PLAIN_IMPLEMENTATION
        )
    else:
        with pytest.raises(ValueError, match="Ascend950PR"):
            operator._resolve_implementation("npu:3", "default")


def test_mxfp8_provider_is_formal_only_on_ascend_950pr(monkeypatch):
    operator = _mxfp8_operator()
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "Ascend950PR_957b",
    )
    monkeypatch.setattr(
        operator,
        "_runtime_supported",
        lambda: True,
        raising=False,
    )

    assert operator.get_formal_implementations("npu:1") == [
        MXFP8_IMPLEMENTATION
    ]
    assert operator.get_formal_implementations("cuda:0") == []

    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "Ascend910C",
    )
    assert operator.get_formal_implementations("npu:1") == []


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("Ascend950PR_957b", True),
        ("Ascend950PR", True),
        ("Ascend950DT", False),
        ("Ascend910C", False),
        ("NVIDIA H20", False),
    ],
)
def test_mxfp4_provider_is_formal_only_on_ascend_950pr(
    monkeypatch,
    device_name,
    expected,
):
    operator = _mxfp4_operator()
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: device_name,
    )
    monkeypatch.setattr(
        operator,
        "_runtime_supported",
        lambda: True,
        raising=False,
    )

    assert operator.get_formal_implementations("npu:2") == (
        [MXFP4_IMPLEMENTATION] if expected else []
    )
    assert operator.get_formal_implementations("cuda:0") == []
    if expected:
        assert operator._resolve_implementation("npu:2", "default") == (
            MXFP4_IMPLEMENTATION
        )
    else:
        with pytest.raises(ValueError, match="Ascend950PR"):
            operator._resolve_implementation("npu:2", "default")


@pytest.mark.parametrize(
    ("operator_factory", "runtime"),
    [
        (
            _plain_operator,
            SimpleNamespace(npu_quant_matmul=lambda *args, **kwargs: None),
        ),
        (
            _plain_operator,
            SimpleNamespace(npu_dynamic_quant=lambda *args, **kwargs: None),
        ),
        (
            _mxfp8_operator,
            SimpleNamespace(
                npu_quant_matmul=lambda *args, **kwargs: None,
                float8_e8m0fnu=object(),
            ),
        ),
        (
            _mxfp8_operator,
            SimpleNamespace(
                npu_dynamic_mx_quant=lambda *args, **kwargs: None,
                float8_e8m0fnu=object(),
            ),
        ),
        (
            _mxfp4_operator,
            SimpleNamespace(
                npu_quant_matmul=lambda *args, **kwargs: None,
                float4_e2m1fn_x2=296,
                float8_e8m0fnu=293,
            ),
        ),
        (
            _mxfp4_operator,
            SimpleNamespace(
                npu_dynamic_mx_quant=lambda *args, **kwargs: None,
                float4_e2m1fn_x2=296,
                float8_e8m0fnu=293,
            ),
        ),
    ],
)
def test_950pr_linear_formal_gate_requires_quantization_runtime_symbol(
    monkeypatch,
    operator_factory,
    runtime,
):
    operator = operator_factory()
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "Ascend950PR_957b",
    )
    monkeypatch.setattr(operator, "_load_torch_npu", lambda: runtime)

    assert operator.get_formal_implementations("npu:0") == []


@pytest.mark.parametrize(
    "runtime",
    [
        SimpleNamespace(
            npu_dynamic_mx_quant=lambda *args, **kwargs: None,
            npu_quant_matmul=lambda *args, **kwargs: None,
            float8_e8m0fnu=293,
        ),
        SimpleNamespace(
            npu_dynamic_mx_quant=lambda *args, **kwargs: None,
            npu_quant_matmul=lambda *args, **kwargs: None,
            float4_e2m1fn_x2=296,
        ),
    ],
)
def test_950pr_mxfp4_linear_formal_gate_requires_native_dtype_codes(
    monkeypatch,
    runtime,
):
    operator = _mxfp4_operator()
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "Ascend950PR_957b",
    )
    monkeypatch.setattr(operator, "_load_torch_npu", lambda: runtime)

    assert operator.get_formal_implementations("npu:0") == []


def test_950pr_mxfp8_linear_formal_gate_requires_e8m0_dtype(
    monkeypatch,
):
    operator = _mxfp8_operator()
    runtime = SimpleNamespace(
        npu_dynamic_mx_quant=lambda *args, **kwargs: None,
        npu_quant_matmul=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        operator,
        "_npu_device_name",
        lambda device: "Ascend950PR_957b",
    )
    monkeypatch.setattr(operator, "_load_torch_npu", lambda: runtime)
    monkeypatch.setattr(
        torch,
        "float8_e8m0fnu",
        None,
        raising=False,
    )

    assert operator.get_formal_implementations("npu:0") == []


def test_plain_fp8_provider_prepares_kn_weight_and_fp32_scales(monkeypatch):
    operator = _plain_operator()
    quantize_calls = []
    quant_matmul = lambda *args, **kwargs: None

    def fake_dynamic_quant(source, *, dst_type):
        quantize_calls.append((tuple(source.shape), source.dtype, dst_type))
        rows = source.shape[0]
        scale = torch.arange(1, rows + 1, dtype=torch.float32)
        return source.to(dst_type), scale

    monkeypatch.setattr(
        operator,
        "_resolve_implementation",
        lambda device, implementation: PLAIN_IMPLEMENTATION,
    )
    monkeypatch.setattr(
        operator,
        "_dynamic_quant_callable",
        lambda: fake_dynamic_quant,
    )
    monkeypatch.setattr(
        operator,
        "_quant_matmul_callable",
        lambda: quant_matmul,
    )
    data = operator.generate_test_data(
        batch_size=3,
        input_dim=64,
        output_dim=32,
        bias=False,
    )

    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.FP8,
        PLAIN_IMPLEMENTATION,
    )

    assert quantize_calls == [
        ((3, 64), torch.bfloat16, torch.float8_e4m3fn),
        ((32, 64), torch.bfloat16, torch.float8_e4m3fn),
    ]
    assert prepared["op"] is quant_matmul
    assert prepared["A"].shape == (3, 64)
    assert prepared["A"].dtype is torch.float8_e4m3fn
    assert prepared["B"].shape == (64, 32)
    assert prepared["B"].dtype is torch.float8_e4m3fn
    assert prepared["B"].is_contiguous()
    assert prepared["scale_a"].shape == (3,)
    assert prepared["scale_a"].dtype is torch.float32
    assert prepared["scale_b"].shape == (32,)
    assert prepared["scale_b"].dtype is torch.float32
    assert "output" not in prepared
    assert not operator._declares_preallocated_output_contract(
        prepared,
        PLAIN_IMPLEMENTATION,
    )


def test_plain_fp8_provider_executes_only_cached_quant_matmul():
    operator = _plain_operator()
    calls = []
    a = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
    b = torch.empty(64, 32, dtype=torch.float8_e4m3fn)
    scale_a = torch.ones(2, dtype=torch.float32)
    scale_b = torch.ones(32, dtype=torch.float32)
    expected = torch.empty(2, 32, dtype=torch.bfloat16)

    def fake_quant_matmul(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    prepared = {
        "implementation": PLAIN_IMPLEMENTATION,
        "op": fake_quant_matmul,
        "A": a,
        "B": b,
        "scale_a": scale_a,
        "scale_b": scale_b,
    }

    result = operator._execute_core_operator(
        prepared,
        PLAIN_IMPLEMENTATION,
    )

    assert result is expected
    assert calls == [
        (
            (a, b, scale_b),
            {
                "pertoken_scale": scale_a,
                "bias": None,
                "output_dtype": torch.bfloat16,
            },
        )
    ]


def test_mxfp8_provider_prepares_group32_pair_packed_scale_layout(
    monkeypatch,
):
    operator = _mxfp8_operator()
    quantize_calls = []
    e8m0_dtype = object()
    quant_matmul = lambda *args, **kwargs: None

    def fake_dynamic_mx_quant(
        source,
        *,
        dst_type,
        block_size,
        round_mode,
    ):
        quantize_calls.append(
            (
                tuple(source.shape),
                source.dtype,
                dst_type,
                block_size,
                round_mode,
            )
        )
        rows, width = source.shape
        scale = torch.arange(
            rows * (width // 64) * 2,
            dtype=torch.uint8,
        ).reshape(rows, width // 64, 2)
        return source.to(dst_type), scale

    monkeypatch.setattr(
        operator,
        "_resolve_implementation",
        lambda device, implementation: MXFP8_IMPLEMENTATION,
    )
    monkeypatch.setattr(
        operator,
        "_dynamic_mx_quant_callable",
        lambda: fake_dynamic_mx_quant,
    )
    monkeypatch.setattr(
        operator,
        "_quant_matmul_callable",
        lambda: quant_matmul,
    )
    monkeypatch.setattr(
        operator,
        "_e8m0_scale_dtype",
        lambda: e8m0_dtype,
    )
    data = operator.generate_test_data(
        batch_size=3,
        input_dim=128,
        output_dim=32,
        bias=False,
    )

    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.MXFP8,
        MXFP8_IMPLEMENTATION,
    )

    assert quantize_calls == [
        (
            (3, 128),
            torch.bfloat16,
            torch.float8_e4m3fn,
            32,
            "rint",
        ),
        (
            (32, 128),
            torch.bfloat16,
            torch.float8_e4m3fn,
            32,
            "rint",
        ),
    ]
    assert prepared["op"] is quant_matmul
    assert prepared["A"].shape == (3, 128)
    assert prepared["A"].dtype is torch.float8_e4m3fn
    assert prepared["B"].shape == (128, 32)
    assert prepared["B"].dtype is torch.float8_e4m3fn
    assert prepared["B"].is_contiguous()
    assert prepared["scale_a"].shape == (3, 2, 2)
    assert prepared["scale_a"].dtype is torch.uint8
    assert prepared["scale_b"].shape == (2, 32, 2)
    assert prepared["scale_b"].dtype is torch.uint8
    assert prepared["scale_b"].is_contiguous()
    assert prepared["scale_dtype"] is e8m0_dtype
    assert prepared["group_size"] == 32
    assert "output" not in prepared


def test_mxfp8_provider_executes_cached_group32_quant_matmul():
    operator = _mxfp8_operator()
    calls = []
    e8m0_dtype = object()
    a = torch.empty(2, 128, dtype=torch.float8_e4m3fn)
    b = torch.empty(128, 32, dtype=torch.float8_e4m3fn)
    scale_a = torch.zeros(2, 2, 2, dtype=torch.uint8)
    scale_b = torch.zeros(2, 32, 2, dtype=torch.uint8)
    expected = torch.empty(2, 32, dtype=torch.bfloat16)

    def fake_quant_matmul(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    prepared = {
        "implementation": MXFP8_IMPLEMENTATION,
        "op": fake_quant_matmul,
        "A": a,
        "B": b,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "scale_dtype": e8m0_dtype,
        "group_size": 32,
    }

    result = operator._execute_core_operator(
        prepared,
        MXFP8_IMPLEMENTATION,
    )

    assert result is expected
    assert calls == [
        (
            (a, b, scale_b),
            {
                "pertoken_scale": scale_a,
                "bias": None,
                "output_dtype": torch.bfloat16,
                "scale_dtype": e8m0_dtype,
                "pertoken_scale_dtype": e8m0_dtype,
                "group_sizes": [1, 1, 32],
            },
        )
    ]


def test_mxfp8_provider_rejects_k_without_complete_e8m0_scale_pairs(
    monkeypatch,
):
    operator = _mxfp8_operator()
    monkeypatch.setattr(
        operator,
        "_resolve_implementation",
        lambda device, implementation: MXFP8_IMPLEMENTATION,
    )
    data = operator.generate_test_data(
        batch_size=2,
        input_dim=96,
        output_dim=32,
        bias=False,
    )

    with pytest.raises(ValueError, match="K.*64"):
        operator._prepare_data_for_core_operator(
            data,
            "cpu",
            PrecisionType.MXFP8,
            MXFP8_IMPLEMENTATION,
        )


def test_mxfp4_provider_prepares_native_packed_storage_and_exact_quant_kwargs(
    monkeypatch,
):
    operator = _mxfp4_operator()
    quantize_calls = []
    quant_matmul = lambda *args, **kwargs: None

    def fake_dynamic_mx_quant(
        source,
        *,
        dst_type,
        block_size,
        round_mode,
    ):
        quantize_calls.append(
            (
                tuple(source.shape),
                source.dtype,
                dst_type,
                block_size,
                round_mode,
            )
        )
        packed = torch.empty(
            *source.shape[:-1],
            source.shape[-1] // 2,
            dtype=torch.uint8,
        )
        scale = torch.empty(
            *source.shape[:-1],
            source.shape[-1] // 64,
            2,
            dtype=torch.uint8,
        )
        return packed, scale

    monkeypatch.setattr(
        operator,
        "_resolve_implementation",
        lambda device, implementation: MXFP4_IMPLEMENTATION,
    )
    monkeypatch.setattr(
        operator,
        "_dynamic_mx_quant_callable",
        lambda: fake_dynamic_mx_quant,
    )
    monkeypatch.setattr(
        operator,
        "_quant_matmul_callable",
        lambda: quant_matmul,
    )
    data = operator.generate_test_data(
        batch_size=16,
        input_dim=64,
        output_dim=32,
        bias=False,
    )

    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.MXFP4,
        MXFP4_IMPLEMENTATION,
    )

    assert quantize_calls == [
        ((16, 64), torch.bfloat16, 296, 32, "round"),
        ((32, 64), torch.bfloat16, 296, 32, "round"),
    ]
    assert prepared["op"] is quant_matmul
    assert prepared["A"].shape == (16, 32)
    assert prepared["A"].dtype is torch.uint8
    assert prepared["B"].shape == (32, 32)
    assert prepared["B"].dtype is torch.uint8
    assert prepared["scale_a"].shape == (16, 1, 2)
    assert prepared["scale_a"].dtype is torch.uint8
    assert prepared["scale_b"].shape == (1, 32, 2)
    assert prepared["scale_b"].dtype is torch.uint8
    assert prepared["x_dtype"] == 296
    assert prepared["scale_dtype"] == 293
    assert "input" not in prepared
    assert "weight" not in prepared
    assert "output" not in prepared


def test_mxfp4_weight_and_scale_preserve_corresponding_transpose_views(
    monkeypatch,
):
    operator = _mxfp4_operator()
    quantized_outputs = []

    def fake_dynamic_mx_quant(
        source,
        *,
        dst_type,
        block_size,
        round_mode,
    ):
        del dst_type, block_size, round_mode
        packed = torch.empty(
            *source.shape[:-1],
            source.shape[-1] // 2,
            dtype=torch.uint8,
        )
        packed.flatten().copy_(
            torch.arange(packed.numel(), dtype=torch.int64)
            .remainder(251)
            .to(torch.uint8)
        )
        scale = torch.empty(
            *source.shape[:-1],
            source.shape[-1] // 64,
            2,
            dtype=torch.uint8,
        )
        scale.flatten().copy_(
            torch.arange(scale.numel(), dtype=torch.int64)
            .remainder(251)
            .to(torch.uint8)
        )
        quantized_outputs.append((packed, scale))
        return packed, scale

    monkeypatch.setattr(
        operator,
        "_resolve_implementation",
        lambda device, implementation: MXFP4_IMPLEMENTATION,
    )
    monkeypatch.setattr(
        operator,
        "_dynamic_mx_quant_callable",
        lambda: fake_dynamic_mx_quant,
    )
    monkeypatch.setattr(
        operator,
        "_quant_matmul_callable",
        lambda: lambda *args, **kwargs: None,
    )

    prepared = operator._prepare_data_for_core_operator(
        operator.generate_test_data(
            batch_size=16,
            input_dim=64,
            output_dim=32,
            bias=False,
        ),
        "cpu",
        PrecisionType.MXFP4,
        MXFP4_IMPLEMENTATION,
    )
    weight_packed, weight_scale = quantized_outputs[1]
    expected_weight_view = weight_packed.transpose(0, 1)
    expected_scale_view = weight_scale.transpose(0, 1)

    assert prepared["B"].shape == (32, 32)
    assert prepared["B"].stride() == (1, 32)
    assert prepared["B"]._base is weight_packed
    assert (
        prepared["B"].untyped_storage().data_ptr()
        == weight_packed.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(prepared["B"], expected_weight_view)

    assert prepared["scale_b"].shape == (1, 32, 2)
    assert prepared["scale_b"].stride() == (2, 2, 1)
    assert prepared["scale_b"]._base is weight_scale
    assert (
        prepared["scale_b"].untyped_storage().data_ptr()
        == weight_scale.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(prepared["scale_b"], expected_scale_view)


def test_mxfp4_timed_dispatch_is_one_native_call_and_retains_bf16_output():
    from operator_test_framework import OperatorTestFramework

    operator = _mxfp4_operator()
    calls = []
    a = torch.empty(16, 32, dtype=torch.uint8)
    b = torch.empty(32, 32, dtype=torch.uint8)
    scale_a = torch.empty(16, 1, 2, dtype=torch.uint8)
    scale_b = torch.empty(1, 32, 2, dtype=torch.uint8)
    expected = torch.empty(16, 32, dtype=torch.bfloat16)

    def fake_quant_matmul(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    prepared = {
        "implementation": MXFP4_IMPLEMENTATION,
        "op": fake_quant_matmul,
        "A": a,
        "B": b,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "x_dtype": 296,
        "scale_dtype": 293,
    }
    retained_outputs = [None]

    OperatorTestFramework._dispatch_prepared_payloads(
        [prepared],
        operator._execute_core_operator,
        MXFP4_IMPLEMENTATION,
        retained_outputs,
    )

    assert retained_outputs == [expected]
    assert retained_outputs[0].dtype is torch.bfloat16
    assert calls == [
        (
            (a, b, scale_b),
            {
                "pertoken_scale": scale_a,
                "output_dtype": torch.bfloat16,
                "x1_dtype": 296,
                "x2_dtype": 296,
                "scale_dtype": 293,
                "pertoken_scale_dtype": 293,
                "group_sizes": [1, 1, 32],
            },
        )
    ]


def test_mxfp4_prepared_payloads_have_fresh_physical_storage(monkeypatch):
    from operator_test_framework import OperatorTestFramework

    operator = _mxfp4_operator()

    def fake_dynamic_mx_quant(
        source,
        *,
        dst_type,
        block_size,
        round_mode,
    ):
        del dst_type, block_size, round_mode
        return (
            torch.empty(
                *source.shape[:-1],
                source.shape[-1] // 2,
                dtype=torch.uint8,
            ),
            torch.empty(
                *source.shape[:-1],
                source.shape[-1] // 64,
                2,
                dtype=torch.uint8,
            ),
        )

    monkeypatch.setattr(
        operator,
        "_resolve_implementation",
        lambda device, implementation: MXFP4_IMPLEMENTATION,
    )
    monkeypatch.setattr(
        operator,
        "_dynamic_mx_quant_callable",
        lambda: fake_dynamic_mx_quant,
    )
    monkeypatch.setattr(
        operator,
        "_quant_matmul_callable",
        lambda: lambda *args, **kwargs: torch.empty(
            16, 32, dtype=torch.bfloat16
        ),
    )
    data = operator.generate_test_data(
        batch_size=16,
        input_dim=64,
        output_dim=32,
        bias=False,
    )
    prepared_payloads = [
        operator._prepare_data_for_core_operator(
            data,
            "cpu",
            PrecisionType.MXFP4,
            MXFP4_IMPLEMENTATION,
        )
        for _ in range(3)
    ]
    input_sets = [
        OperatorTestFramework._prepared_input_values(prepared)
        for prepared in prepared_payloads
    ]

    assert OperatorTestFramework._verify_independent_storage_sets(
        input_sets,
        "cpu",
        "MXFP4 Linear packed input",
    ) == (3, 12)
