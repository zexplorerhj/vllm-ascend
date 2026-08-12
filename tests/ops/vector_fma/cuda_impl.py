"""H20-native scalar and packed inline-PTX Vector FMA providers."""

from __future__ import annotations

from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.cpp_extension import load as _cpp_extension_load

from operator_test_framework import PrecisionType
from vector_fma.base import (
    VectorFmaConfig,
    VectorFmaOperatorTestBase,
)


H20_DEVICE_NAME = "NVIDIA H20-3e"
FP16_SCALAR_PROVIDER = "cuda_ptx_f16_scalar_fma"
FP16_X2_PROVIDER = "cuda_ptx_f16x2_fma"
BF16_SCALAR_PROVIDER = "cuda_ptx_bf16_scalar_fma"
BF16_X2_PROVIDER = "cuda_ptx_bf16x2_fma"

_SUPPORTED_ACCUMULATORS = (4, 8, 16)
_SUPPORTED_LANE_WIDTHS = (1, 2)

DEFAULT_CUDA_SCALAR_VECTOR_FMA_CONFIG = VectorFmaConfig(
    elements=78 * 256,
    fma_depth=65536,
    accumulators=8,
    block_size=256,
    num_programs=78,
)
DEFAULT_CUDA_X2_VECTOR_FMA_CONFIG = VectorFmaConfig(
    elements=78 * 256 * 2,
    fma_depth=65536,
    accumulators=8,
    block_size=256,
    num_programs=78,
)


def _validate_instruction_lane_width(instruction_lane_width: int) -> None:
    if instruction_lane_width not in _SUPPORTED_LANE_WIDTHS:
        raise ValueError(
            "instruction_lane_width must be 1 or 2, got "
            f"{instruction_lane_width!r}"
        )


def validate_cuda_launch_geometry(
    config: VectorFmaConfig,
    instruction_lane_width: int,
) -> None:
    """Require a one-to-one mapping from CUDA threads to scalar lanes."""
    _validate_instruction_lane_width(instruction_lane_width)
    if config.accumulators not in _SUPPORTED_ACCUMULATORS:
        raise ValueError(
            "accumulators must be one of "
            f"{_SUPPORTED_ACCUMULATORS}, got {config.accumulators}"
        )
    if config.block_size > 1024:
        raise ValueError(
            f"block_size must not exceed CUDA's 1024-thread limit, got "
            f"{config.block_size}"
        )
    physical_scalar_lanes = (
        config.block_size
        * config.num_programs
        * instruction_lane_width
    )
    if physical_scalar_lanes != config.elements:
        raise ValueError(
            "block_size * num_programs * instruction_lane_width must equal "
            "the global scalar lanes exactly: "
            f"{physical_scalar_lanes} != {config.elements}"
        )


def cuda_ptx_fma_instruction_count(
    config: VectorFmaConfig,
    instruction_lane_width: int,
) -> int:
    """Return issued scalar or x2 PTX FMA instructions for the workload."""
    _validate_instruction_lane_width(instruction_lane_width)
    if config.elements % instruction_lane_width:
        raise ValueError(
            "global scalar lanes must be divisible by "
            f"instruction_lane_width={instruction_lane_width}"
        )
    return (
        config.elements
        // instruction_lane_width
        * config.accumulators
        * config.fma_depth
    )


def cuda_ptx_flops(
    config: VectorFmaConfig,
    instruction_lane_width: int,
) -> int:
    """Count two FLOPs per scalar lane, including both lanes of x2 PTX."""
    instructions = cuda_ptx_fma_instruction_count(
        config,
        instruction_lane_width,
    )
    return instructions * instruction_lane_width * 2


def _cuda_source_path() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "vector_fma_cuda.cu"


@lru_cache(maxsize=1)
def _load_cuda_extension() -> Any:
    """Build the SM90 extension once, entirely before Event timing."""
    return _cpp_extension_load(
        name="vector_fma_cuda_ptx_sm90",
        sources=[str(_cuda_source_path())],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--std=c++17",
            "--generate-code=arch=compute_90,code=sm_90",
            "--ptxas-options=-v",
        ],
        verbose=True,
    )


def _get_cuda_device_name(device: str) -> str:
    return torch.cuda.get_device_name(torch.device(device))


class CudaPtxVectorFmaOperatorTest(VectorFmaOperatorTestBase):
    """One cached H20 inline-PTX launch into a preallocated output tensor."""

    def __init__(
        self,
        precision: PrecisionType,
        packed: bool,
        config: Optional[VectorFmaConfig] = None,
    ) -> None:
        if not isinstance(packed, bool):
            raise ValueError("packed must be a bool")
        if config is None:
            config = (
                DEFAULT_CUDA_X2_VECTOR_FMA_CONFIG
                if packed
                else DEFAULT_CUDA_SCALAR_VECTOR_FMA_CONFIG
            )
        super().__init__(precision, config)
        self.packed = packed
        self.instruction_lane_width = 2 if packed else 1
        self.flops_per_instruction = 2 * self.instruction_lane_width
        if precision is PrecisionType.FP16:
            self.provider_name = (
                FP16_X2_PROVIDER if packed else FP16_SCALAR_PROVIDER
            )
            self.launch_name = "launch_f16x2" if packed else "launch_f16_scalar"
        else:
            self.provider_name = (
                BF16_X2_PROVIDER if packed else BF16_SCALAR_PROVIDER
            )
            self.launch_name = (
                "launch_bf16x2" if packed else "launch_bf16_scalar"
            )
        validate_cuda_launch_geometry(
            self.config,
            self.instruction_lane_width,
        )

    def _require_h20_device(self, device: str) -> None:
        if not device.startswith("cuda"):
            raise RuntimeError(
                f"{self.provider_name} requires a CUDA device, got {device!r}"
            )
        try:
            device_name = _get_cuda_device_name(device)
        except Exception as exc:
            raise RuntimeError(
                f"unable to query CUDA device {device!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if device_name != H20_DEVICE_NAME:
            raise RuntimeError(
                f"{self.provider_name} requires exact CUDA device "
                f"{H20_DEVICE_NAME!r}, got {device_name!r}"
            )

    def get_available_implementations(self, device: str) -> List[str]:
        if not device.startswith("cuda"):
            return []
        try:
            self._require_h20_device(device)
        except RuntimeError:
            return []
        return [self.provider_name]

    def _select_implementation(
        self,
        prepared_data: Dict[str, Any],
        implementation: str,
    ) -> str:
        selected = (
            prepared_data.get("implementation", self.provider_name)
            if implementation == "default"
            else implementation
        )
        if selected != self.provider_name:
            raise ValueError(
                f"implementation {selected!r} does not match "
                f"{self.provider_name!r}"
            )
        return selected

    def _bind_cached_launch(
        self,
        module: Any,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        config: VectorFmaConfig,
    ) -> Any:
        try:
            launch = getattr(module, self.launch_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"CUDA extension has no {self.launch_name!r} entry point"
            ) from exc
        return partial(
            launch,
            a,
            b,
            output,
            config.elements,
            config.fma_depth,
            config.accumulators,
            config.block_size,
            config.num_programs,
        )

    def _prepare_data_for_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Dict[str, Any]:
        selected = (
            self.provider_name
            if implementation == "default"
            else implementation
        )
        if selected != self.provider_name:
            raise ValueError(
                f"implementation {selected!r} does not match "
                f"{self.provider_name!r}"
            )
        self._require_h20_device(device)
        config = self._data_config(data)
        validate_cuda_launch_geometry(
            config,
            self.instruction_lane_width,
        )

        # Extension compilation and symbol resolution happen before the
        # framework can enter its Event-timed execution callback.
        module = _load_cuda_extension()
        prepared = super()._prepare_data_for_core_operator(
            data,
            device,
            precision,
            selected,
        )
        prepared["metadata"].update(
            {
                "instruction_form": (
                    "packed_x2" if self.packed else "scalar"
                ),
                "instruction_lane_width": self.instruction_lane_width,
                "flops_per_instruction": self.flops_per_instruction,
                "ptx_fma_instructions": cuda_ptx_fma_instruction_count(
                    config,
                    self.instruction_lane_width,
                ),
            }
        )
        prepared["module"] = module
        prepared["kernel"] = self._bind_cached_launch(
            module,
            prepared["a"],
            prepared["b"],
            prepared["output"],
            config,
        )
        return prepared

    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> torch.Tensor:
        self._select_implementation(prepared_data, implementation)
        prepared_data["kernel"]()
        return prepared_data["output"]

    def _declares_preallocated_output_contract(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> bool:
        selected = (
            prepared_data.get("implementation", self.provider_name)
            if implementation == "default"
            else implementation
        )
        return selected == self.provider_name
