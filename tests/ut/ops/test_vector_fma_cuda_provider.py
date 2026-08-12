from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import PrecisionType  # noqa: E402
from vector_fma.base import VectorFmaConfig  # noqa: E402
from vector_fma.cuda_impl import (  # noqa: E402
    CudaPtxVectorFmaOperatorTest,
    _load_cuda_extension,
    cuda_ptx_fma_instruction_count,
    cuda_ptx_flops,
    validate_cuda_launch_geometry,
)


def _config(
    *,
    elements=16,
    fma_depth=32,
    accumulators=4,
    block_size=8,
    num_programs=2,
):
    return VectorFmaConfig(
        elements=elements,
        fma_depth=fma_depth,
        accumulators=accumulators,
        block_size=block_size,
        num_programs=num_programs,
    )


@pytest.mark.parametrize(
    ("precision", "packed", "provider", "launch_name", "lane_width"),
    [
        (
            PrecisionType.FP16,
            False,
            "cuda_ptx_f16_scalar_fma",
            "launch_f16_scalar",
            1,
        ),
        (
            PrecisionType.FP16,
            True,
            "cuda_ptx_f16x2_fma",
            "launch_f16x2",
            2,
        ),
        (
            PrecisionType.BF16,
            False,
            "cuda_ptx_bf16_scalar_fma",
            "launch_bf16_scalar",
            1,
        ),
        (
            PrecisionType.BF16,
            True,
            "cuda_ptx_bf16x2_fma",
            "launch_bf16x2",
            2,
        ),
    ],
)
def test_precision_and_instruction_form_select_one_provider(
    precision,
    packed,
    provider,
    launch_name,
    lane_width,
):
    config = _config(elements=16 * lane_width)
    operator = CudaPtxVectorFmaOperatorTest(precision, packed, config)

    assert operator.provider_name == provider
    assert operator.launch_name == launch_name
    assert operator.instruction_lane_width == lane_width
    assert operator.flops_per_instruction == 2 * lane_width


@pytest.mark.parametrize("accumulators", [4, 8, 16])
def test_scalar_geometry_maps_every_global_scalar_lane_once(accumulators):
    validate_cuda_launch_geometry(
        _config(accumulators=accumulators),
        instruction_lane_width=1,
    )


def test_x2_geometry_counts_elements_as_scalar_lanes():
    config = _config(elements=32)

    validate_cuda_launch_geometry(config, instruction_lane_width=2)

    assert cuda_ptx_fma_instruction_count(config, 2) == 32 * 4 * 32 // 2
    assert cuda_ptx_flops(config, 2) == 2 * 32 * 4 * 32


def test_x2_flop_accounting_does_not_count_packed_lanes_twice():
    config = _config(elements=32)
    operator = CudaPtxVectorFmaOperatorTest(
        PrecisionType.FP16,
        True,
        config,
    )
    data = {"config": config}

    assert operator.calculate_tflops(data, avg_time_ms=1.0) == pytest.approx(
        (2 * 32 * 4 * 32) / 1e9
    )


@pytest.mark.parametrize(
    ("lane_width", "elements"),
    [(1, 15), (1, 17), (2, 31), (2, 33)],
)
def test_geometry_rejects_physical_threads_that_do_not_match_scalar_lanes(
    lane_width,
    elements,
):
    with pytest.raises(ValueError, match="global scalar lanes"):
        validate_cuda_launch_geometry(
            _config(elements=elements),
            instruction_lane_width=lane_width,
        )


@pytest.mark.parametrize("accumulators", [1, 6, 32])
def test_geometry_rejects_unsupported_accumulator_count(accumulators):
    with pytest.raises(ValueError, match="accumulators"):
        validate_cuda_launch_geometry(
            _config(accumulators=accumulators),
            instruction_lane_width=1,
        )


def test_only_the_exact_h20_3e_device_name_is_advertised(monkeypatch):
    operator = CudaPtxVectorFmaOperatorTest(
        PrecisionType.BF16,
        True,
        _config(elements=32),
    )
    monkeypatch.setattr(
        "vector_fma.cuda_impl._get_cuda_device_name",
        lambda device: "NVIDIA H20-3e",
    )

    assert operator.get_available_implementations("cuda:0") == [
        "cuda_ptx_bf16x2_fma"
    ]

    for wrong_name in ("NVIDIA H20", "NVIDIA H20 NVL", "NVIDIA H200"):
        monkeypatch.setattr(
            "vector_fma.cuda_impl._get_cuda_device_name",
            lambda device, name=wrong_name: name,
        )
        assert operator.get_available_implementations("cuda:0") == []
        with pytest.raises(RuntimeError, match="requires exact CUDA device"):
            operator._require_h20_device("cuda:0")


def test_non_cuda_device_is_rejected_without_querying_cuda(monkeypatch):
    get_name = Mock(side_effect=AssertionError("must not query CUDA"))
    monkeypatch.setattr(
        "vector_fma.cuda_impl._get_cuda_device_name",
        get_name,
    )
    operator = CudaPtxVectorFmaOperatorTest(
        PrecisionType.FP16,
        False,
        _config(),
    )

    assert operator.get_available_implementations("cpu") == []
    get_name.assert_not_called()


def test_extension_build_is_cached_and_uses_sm90_flags(monkeypatch):
    module = object()
    loader = Mock(return_value=module)
    monkeypatch.setattr("vector_fma.cuda_impl._cpp_extension_load", loader)
    _load_cuda_extension.cache_clear()

    try:
        assert _load_cuda_extension() is module
        assert _load_cuda_extension() is module
    finally:
        _load_cuda_extension.cache_clear()

    loader.assert_called_once()
    kwargs = loader.call_args.kwargs
    assert kwargs["name"] == "vector_fma_cuda_ptx_sm90"
    assert kwargs["sources"] == [
        str(OPS_ROOT / "vector_fma" / "csrc" / "vector_fma_cuda.cu")
    ]
    assert "-O3" in kwargs["extra_cuda_cflags"]
    assert "-lineinfo" in kwargs["extra_cuda_cflags"]
    assert any("compute_90" in flag for flag in kwargs["extra_cuda_cflags"])
    assert any("sm_90" in flag for flag in kwargs["extra_cuda_cflags"])


def test_cached_module_binding_selects_one_out_launch():
    launch = Mock()
    module = SimpleNamespace(launch_f16x2=launch)
    operator = CudaPtxVectorFmaOperatorTest(
        PrecisionType.FP16,
        True,
        _config(elements=32),
    )
    a = torch.empty((4, 32), dtype=torch.float16)
    b = torch.empty_like(a)
    output = torch.empty_like(a)

    cached_launch = operator._bind_cached_launch(
        module,
        a,
        b,
        output,
        operator.config,
    )
    cached_launch()

    launch.assert_called_once_with(a, b, output, 32, 32, 4, 8, 2)


def test_cached_launch_uses_the_prepared_data_config_not_constructor_config():
    launch = Mock()
    module = SimpleNamespace(launch_f16_scalar=launch)
    operator = CudaPtxVectorFmaOperatorTest(
        PrecisionType.FP16,
        False,
        _config(elements=16),
    )
    prepared_config = _config(
        elements=32,
        fma_depth=64,
        block_size=8,
        num_programs=4,
    )
    a = torch.empty((4, 32), dtype=torch.float16)
    b = torch.empty_like(a)
    output = torch.empty_like(a)

    cached_launch = operator._bind_cached_launch(
        module,
        a,
        b,
        output,
        prepared_config,
    )
    cached_launch()

    launch.assert_called_once_with(a, b, output, 32, 64, 4, 8, 4)


def test_execute_calls_one_cached_module_launch_and_returns_output_alias():
    operator = CudaPtxVectorFmaOperatorTest(
        PrecisionType.BF16,
        True,
        _config(elements=32),
    )
    output = torch.empty((4, 32), dtype=torch.bfloat16)
    launch = Mock()
    prepared = {
        "kernel": launch,
        "output": output,
        "implementation": "cuda_ptx_bf16x2_fma",
    }

    result = operator._execute_core_operator(
        prepared,
        "cuda_ptx_bf16x2_fma",
    )

    launch.assert_called_once_with()
    assert result is output


def test_wrong_provider_is_rejected_before_cached_launch():
    operator = CudaPtxVectorFmaOperatorTest(
        PrecisionType.FP16,
        False,
        _config(),
    )
    launch = Mock()
    prepared = {
        "kernel": launch,
        "output": torch.empty((4, 16), dtype=torch.float16),
        "implementation": "cuda_ptx_f16_scalar_fma",
    }

    with pytest.raises(ValueError, match="does not match"):
        operator._execute_core_operator(
            prepared,
            "cuda_ptx_bf16_scalar_fma",
        )
    launch.assert_not_called()


def test_preallocated_output_contract_is_declared_only_for_own_provider():
    operator = CudaPtxVectorFmaOperatorTest(
        PrecisionType.FP16,
        True,
        _config(elements=32),
    )

    assert operator._declares_preallocated_output_contract(
        {"implementation": "cuda_ptx_f16x2_fma"},
        "cuda_ptx_f16x2_fma",
    )
    assert not operator._declares_preallocated_output_contract(
        {"implementation": "cuda_ptx_f16x2_fma"},
        "cuda_ptx_f16_scalar_fma",
    )
