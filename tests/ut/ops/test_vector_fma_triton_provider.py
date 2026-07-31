from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import PrecisionType  # noqa: E402
from vector_fma.base import VectorFmaConfig  # noqa: E402
from vector_fma.triton_impl import (  # noqa: E402
    DEFAULT_TRITON_VECTOR_FMA_CONFIG,
    TritonVectorFmaOperatorTest,
    _launch_vector_fma_once,
    validate_triton_launch_geometry,
)


def _config(
    *,
    elements=16,
    accumulators=4,
    block_size=8,
    num_programs=2,
):
    return VectorFmaConfig(
        elements=elements,
        fma_depth=32,
        accumulators=accumulators,
        block_size=block_size,
        num_programs=num_programs,
    )


@pytest.mark.parametrize(
    ("precision", "provider"),
    [
        (PrecisionType.FP16, "triton_common_fp16_fma"),
        (PrecisionType.BF16, "triton_common_bf16_fma"),
    ],
)
def test_provider_name_is_bound_to_precision(precision, provider):
    operator = TritonVectorFmaOperatorTest(precision, _config())

    assert operator.provider_name == provider


def test_constructor_has_a_valid_default_config():
    operator = TritonVectorFmaOperatorTest(PrecisionType.FP16)

    assert operator.config == DEFAULT_TRITON_VECTOR_FMA_CONFIG
    validate_triton_launch_geometry(operator.config)


def test_execute_launches_one_cached_raw_kernel():
    operator = TritonVectorFmaOperatorTest(PrecisionType.BF16, _config())
    output = torch.empty((4, 16), dtype=torch.bfloat16)
    fake_kernel = Mock()
    prepared = {
        "kernel": fake_kernel,
        "output": output,
        "implementation": "triton_common_bf16_fma",
    }

    result = operator._execute_core_operator(
        prepared,
        "triton_common_bf16_fma",
    )

    fake_kernel.assert_called_once_with()
    assert result is output


def test_prepared_output_contract_is_declared():
    operator = TritonVectorFmaOperatorTest(PrecisionType.FP16, _config())

    assert operator._declares_preallocated_output_contract(
        {"implementation": "triton_common_fp16_fma"},
        "triton_common_fp16_fma",
    )


@pytest.mark.parametrize("accumulators", [4, 8, 16])
def test_launch_geometry_accepts_all_independent_chain_counts(accumulators):
    validate_triton_launch_geometry(
        _config(accumulators=accumulators),
    )


@pytest.mark.parametrize("elements", [15, 17])
def test_launch_geometry_rejects_any_non_exact_global_lane_mapping(elements):
    with pytest.raises(ValueError, match="exactly equal"):
        validate_triton_launch_geometry(
            _config(elements=elements, block_size=8, num_programs=2),
        )


def test_launch_geometry_rejects_non_power_of_two_chain_count():
    with pytest.raises(ValueError, match="accumulators"):
        validate_triton_launch_geometry(_config(accumulators=6))


def test_launch_geometry_rejects_non_power_of_two_block_size():
    with pytest.raises(ValueError, match="block_size"):
        validate_triton_launch_geometry(
            _config(elements=18, block_size=6, num_programs=3),
        )


def test_unavailable_triton_is_fail_closed(monkeypatch):
    operator = TritonVectorFmaOperatorTest(PrecisionType.FP16, _config())
    monkeypatch.setattr(
        "vector_fma.triton_impl._triton_capability_error",
        lambda: "triton import failed",
    )

    assert operator.get_available_implementations("cuda:0") == []
    with pytest.raises(RuntimeError, match="triton import failed"):
        operator._require_runtime("cuda:0")


def test_config_override_cannot_bypass_geometry_validation(monkeypatch):
    operator = TritonVectorFmaOperatorTest(PrecisionType.FP16, _config())
    data = operator.generate_test_data(
        config=_config(elements=15),
    )
    monkeypatch.setattr(operator, "_require_runtime", lambda device: None)

    with pytest.raises(ValueError, match="exactly equal"):
        operator._prepare_data_for_core_operator(
            data,
            "cpu",
            PrecisionType.FP16,
            operator.provider_name,
        )


def test_compile_rejection_is_persisted_as_unsupported(monkeypatch):
    operator = TritonVectorFmaOperatorTest(PrecisionType.FP16, _config())
    operator._unsupported_reasons.clear()
    prepared = {
        "kernel": Mock(side_effect=RuntimeError("backend rejects tl.fma")),
        "config": operator.config,
    }
    monkeypatch.setattr(
        "vector_fma.triton_impl._triton_capability_error",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="backend rejects tl.fma"):
        operator._probe_compilation(
            prepared,
            "cuda:0",
            PrecisionType.FP16,
        )

    assert operator.get_available_implementations("cuda:0") == []
    with pytest.raises(RuntimeError, match="backend rejects tl.fma"):
        operator._require_runtime("cuda:0")


def test_raw_launch_passes_exact_constexpr_workload():
    launch = Mock()

    class FakeKernel:
        def __getitem__(self, grid):
            assert grid == (2,)
            return launch

    config = _config()
    a = object()
    b = object()
    output = object()

    _launch_vector_fma_once(FakeKernel(), a, b, output, config)

    launch.assert_called_once_with(
        a,
        b,
        output,
        16,
        32,
        num_accumulators=4,
        block_size=8,
    )


def test_kernel_signature_marks_workload_dimensions_constexpr():
    source = (
        OPS_ROOT / "vector_fma" / "triton_impl.py"
    ).read_text(encoding="utf-8")

    assert "n_elements: tl.constexpr" in source
    assert "fma_depth: tl.constexpr" in source


def test_wrong_precision_provider_is_rejected_before_launch():
    operator = TritonVectorFmaOperatorTest(PrecisionType.FP16, _config())
    prepared = {
        "kernel": Mock(),
        "output": torch.empty((4, 16), dtype=torch.float16),
        "implementation": "triton_common_fp16_fma",
    }

    with pytest.raises(ValueError, match="does not match"):
        operator._execute_core_operator(
            prepared,
            "triton_common_bf16_fma",
        )
    prepared["kernel"].assert_not_called()
