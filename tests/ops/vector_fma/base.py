"""Platform-independent contract for register-resident vector FMA tests."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from operator_test_framework import BaseOperatorTest, DeviceType, PrecisionType


@dataclass(frozen=True)
class VectorFmaConfig:
    """Vector FMA workload shape.

    ``elements`` is the scalar-lane count per independent accumulator.  The
    total active scalar lanes are therefore ``elements * accumulators``.
    """

    elements: int
    fma_depth: int
    accumulators: int
    block_size: int
    num_programs: int

    def __post_init__(self) -> None:
        for name in (
            "elements",
            "fma_depth",
            "accumulators",
            "block_size",
            "num_programs",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive non-bool int")


DEFAULT_VECTOR_FMA_CONFIG = VectorFmaConfig(
    elements=4096,
    fma_depth=1024,
    accumulators=8,
    block_size=256,
    num_programs=78,
)


def vector_fma_flops(config: VectorFmaConfig) -> int:
    """Return work for every scalar FMA in all independent chains."""
    return 2 * config.elements * config.accumulators * config.fma_depth


def vector_fma_arithmetic_intensity(
    config: VectorFmaConfig,
    element_size: int,
) -> float:
    """Return FLOPs per byte for two input reads and one output transfer."""
    if (
        not isinstance(element_size, int)
        or isinstance(element_size, bool)
        or element_size <= 0
    ):
        raise ValueError("element_size must be a positive non-bool int")
    transferred = config.elements * config.accumulators * element_size * 3
    return vector_fma_flops(config) / transferred


class VectorFmaOperatorTestBase(BaseOperatorTest):
    """Shared inputs, accounting, and prepared-payload contract for FMA."""

    def __init__(
        self,
        precision: PrecisionType,
        config: VectorFmaConfig = DEFAULT_VECTOR_FMA_CONFIG,
    ) -> None:
        if precision not in (PrecisionType.FP16, PrecisionType.BF16):
            raise ValueError("Vector FMA supports only FP16 and BF16")
        if not isinstance(config, VectorFmaConfig):
            raise ValueError("config must be a VectorFmaConfig")
        super().__init__(f"VectorFma_{precision.name}")
        self.precision = precision
        self.config = config
        self.supported_precisions = [precision]
        self.supported_devices = [DeviceType.CPU, DeviceType.GPU, DeviceType.NPU]

    @staticmethod
    def _resolve_config(
        config: Optional[VectorFmaConfig],
        fallback: VectorFmaConfig,
    ) -> VectorFmaConfig:
        if config is None:
            return fallback
        if not isinstance(config, VectorFmaConfig):
            raise ValueError("config must be a VectorFmaConfig")
        return config

    def generate_test_data(
        self,
        seed: int = 0,
        config: Optional[VectorFmaConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate deterministic finite CPU input lanes and accumulator seed."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected Vector FMA data arguments: {unexpected}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an int")
        resolved = self._resolve_config(config, self.config)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        shape = (resolved.accumulators, resolved.elements)

        def bounded_values() -> torch.Tensor:
            return (
                torch.rand(shape, generator=generator, dtype=torch.float32)
                * 0.25
                + 0.125
            ).to(self.precision.value)

        return {
            "a": bounded_values(),
            "b": bounded_values(),
            "output": bounded_values(),
            "config": resolved,
            "metadata": {
                "elements_per_accumulator": resolved.elements,
                "accumulators": resolved.accumulators,
                "active_scalar_lanes": (
                    resolved.elements * resolved.accumulators
                ),
                "fma_depth": resolved.fma_depth,
                "block_size": resolved.block_size,
                "num_programs": resolved.num_programs,
                "precision": self.precision.name,
                "seed": seed,
            },
        }

    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """Compute the same scalar-lane recurrence without mutating input data."""
        config = self._data_config(data)
        accumulator = data["output"].clone()
        for _ in range(config.fma_depth):
            accumulator = torch.addcmul(accumulator, data["a"], data["b"])
        return accumulator

    def get_available_implementations(self, device: str) -> List[str]:
        del device
        return []

    def _prepare_data_for_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Dict[str, Any]:
        """Copy inputs once and retain configuration beside the cached launch."""
        if precision is not self.precision:
            raise ValueError(
                f"prepared precision {precision.name} does not match "
                f"operator precision {self.precision.name}"
            )
        config = self._data_config(data)
        dtype = precision.value
        prepared = {
            "a": data["a"].to(device=device, dtype=dtype).contiguous(),
            "b": data["b"].to(device=device, dtype=dtype).contiguous(),
            "output": data["output"].to(
                device=device,
                dtype=dtype,
            ).contiguous(),
            "config": config,
            "metadata": dict(data["metadata"]),
            "implementation": implementation,
        }
        prepared["metadata"]["active_scalar_lanes"] = (
            config.elements * config.accumulators
        )
        return prepared

    @abstractmethod
    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> torch.Tensor:
        """Launch exactly one previously cached raw kernel for a payload."""
        raise NotImplementedError

    def run_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> torch.Tensor:
        prepared = self._prepare_data_for_core_operator(
            data,
            device,
            precision,
            implementation,
        )
        return self._execute_core_operator(prepared, implementation)

    def run_device_implementation(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> torch.Tensor:
        return self.run_core_operator(data, device, precision, implementation)

    def calculate_tflops(self, data: Dict[str, Any], avg_time_ms: float) -> float:
        """Calculate teraFLOP/s from the complete scalar-lane workload."""
        if avg_time_ms <= 0:
            return 0.0
        return vector_fma_flops(self._data_config(data)) / (
            avg_time_ms / 1000.0
        ) / 1e12

    @staticmethod
    def _data_config(data: Dict[str, Any]) -> VectorFmaConfig:
        config = data.get("config")
        if not isinstance(config, VectorFmaConfig):
            raise ValueError("data must contain a VectorFmaConfig under 'config'")
        return config
