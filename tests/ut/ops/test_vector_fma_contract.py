from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import PrecisionType  # noqa: E402
from vector_fma.base import (  # noqa: E402
    VectorFmaConfig,
    VectorFmaOperatorTestBase,
    vector_fma_arithmetic_intensity,
    vector_fma_flops,
)


class _PreparedPayloadOperator(VectorFmaOperatorTestBase):
    def get_available_implementations(self, device):
        del device
        return ["test"]

    def _execute_core_operator(self, prepared_data, implementation="default"):
        del implementation
        return prepared_data["output"]


def _config():
    return VectorFmaConfig(
        elements=10,
        fma_depth=5,
        accumulators=4,
        block_size=8,
        num_programs=2,
    )


def test_fma_flops_counts_scalar_lanes():
    config = VectorFmaConfig(
        elements=4096,
        fma_depth=1024,
        accumulators=8,
        block_size=256,
        num_programs=78,
    )
    assert vector_fma_flops(config) == 2 * 4096 * 8 * 1024


def test_fma_config_rejects_bool_and_non_positive_values():
    with pytest.raises(ValueError, match="elements"):
        VectorFmaConfig(
            elements=True,
            fma_depth=1024,
            accumulators=8,
            block_size=256,
            num_programs=78,
        )


@pytest.mark.parametrize(
    "field",
    ["elements", "fma_depth", "accumulators", "block_size", "num_programs"],
)
def test_fma_config_rejects_non_positive_values_for_every_field(field):
    values = {
        "elements": 10,
        "fma_depth": 5,
        "accumulators": 4,
        "block_size": 8,
        "num_programs": 2,
    }
    values[field] = 0

    with pytest.raises(ValueError, match=field):
        VectorFmaConfig(**values)


def test_fma_config_is_immutable():
    config = _config()

    with pytest.raises(FrozenInstanceError):
        config.elements = 11


def test_fma_arithmetic_intensity_uses_all_accumulator_lanes():
    assert vector_fma_arithmetic_intensity(_config(), element_size=2) == 5 / 3


def test_base_generates_deterministic_finite_cpu_lanes_per_accumulator():
    operator = _PreparedPayloadOperator(PrecisionType.FP16, _config())

    first = operator.generate_test_data(seed=17)
    second = operator.generate_test_data(seed=17)

    for name in ("a", "b", "output"):
        assert first[name].device.type == "cpu"
        assert first[name].dtype is torch.float16
        assert first[name].shape == (4, 10)
        assert bool(torch.isfinite(first[name]).all())
        assert torch.equal(first[name], second[name])
    assert first["config"] == _config()
    assert first["metadata"]["active_scalar_lanes"] == 40


def test_base_prepared_payload_exposes_configuration_and_tflops():
    config = _config()
    operator = _PreparedPayloadOperator(PrecisionType.FP16, config)
    data = operator.generate_test_data(seed=17)

    prepared = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        PrecisionType.FP16,
    )

    assert prepared["config"] == config
    assert prepared["metadata"]["elements_per_accumulator"] == 10
    assert operator.calculate_tflops(data, avg_time_ms=2.0) == 2e-7


def test_base_requires_subclasses_to_implement_a_raw_launch():
    with pytest.raises(TypeError, match="_execute_core_operator"):
        VectorFmaOperatorTestBase(PrecisionType.FP16, _config())
