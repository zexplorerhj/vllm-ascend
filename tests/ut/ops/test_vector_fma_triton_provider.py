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
    TritonVectorFmaOperatorTest,
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


def test_launch_geometry_rejects_uncovered_global_lanes():
    with pytest.raises(ValueError, match="cover all global scalar lanes"):
        validate_triton_launch_geometry(
            _config(elements=17, block_size=8, num_programs=2),
        )


def test_launch_geometry_rejects_non_power_of_two_chain_count():
    with pytest.raises(ValueError, match="accumulators"):
        validate_triton_launch_geometry(_config(accumulators=6))


def test_unavailable_triton_is_fail_closed(monkeypatch):
    operator = TritonVectorFmaOperatorTest(PrecisionType.FP16, _config())
    monkeypatch.setattr(
        "vector_fma.triton_impl._triton_capability_error",
        lambda: "triton import failed",
    )

    assert operator.get_available_implementations("cuda:0") == []
    with pytest.raises(RuntimeError, match="triton import failed"):
        operator._require_runtime()


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
