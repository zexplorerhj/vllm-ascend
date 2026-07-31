"""Common Triton provider for the register-resident Vector FMA workload."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional

import torch

from operator_test_framework import PrecisionType
from vector_fma.base import VectorFmaConfig, VectorFmaOperatorTestBase


try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError) as exc:
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _TRITON_IMPORT_ERROR = None


FP16_PROVIDER = "triton_common_fp16_fma"
BF16_PROVIDER = "triton_common_bf16_fma"
_SUPPORTED_ACCUMULATORS = (4, 8, 16)


if triton is not None:

    @triton.jit
    def vector_fma_kernel(
        a_ptr,
        b_ptr,
        output_ptr,
        n_elements,
        fma_depth,
        num_accumulators: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Run all independent chains as one two-dimensional Triton tensor."""
        lanes = (
            tl.program_id(axis=0) * block_size
            + tl.arange(0, block_size)
        )
        chains = tl.arange(0, num_accumulators)
        offsets = chains[:, None] * n_elements + lanes[None, :]
        mask = lanes[None, :] < n_elements

        a = tl.load(a_ptr + offsets, mask=mask, other=0.5)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.25)
        accumulator = tl.load(
            output_ptr + offsets,
            mask=mask,
            other=0.125,
        )
        for _ in tl.range(0, fma_depth, loop_unroll_factor=1):
            accumulator = tl.fma(accumulator, a, b)
        tl.store(output_ptr + offsets, accumulator, mask=mask)

else:
    vector_fma_kernel = None


def _triton_capability_error() -> Optional[str]:
    if _TRITON_IMPORT_ERROR is not None:
        return (
            "Triton is unavailable: "
            f"{type(_TRITON_IMPORT_ERROR).__name__}: "
            f"{_TRITON_IMPORT_ERROR}"
        )
    if tl is None or not hasattr(tl, "fma"):
        return "installed Triton language has no tl.fma"
    if vector_fma_kernel is None:
        return "Triton Vector FMA kernel was not defined"
    return None


def validate_triton_launch_geometry(config: VectorFmaConfig) -> None:
    """Prove the grid covers every global lane without duplicating work."""
    if config.accumulators not in _SUPPORTED_ACCUMULATORS:
        raise ValueError(
            "accumulators must be one of "
            f"{_SUPPORTED_ACCUMULATORS}, got {config.accumulators}"
        )
    capacity = config.block_size * config.num_programs
    if capacity < config.elements:
        raise ValueError(
            "block_size * num_programs must cover all global scalar lanes: "
            f"{capacity} < {config.elements}"
        )


def _launch_vector_fma_once(
    kernel: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    config: VectorFmaConfig,
) -> None:
    kernel[(config.num_programs,)](
        a,
        b,
        output,
        config.elements,
        config.fma_depth,
        num_accumulators=config.accumulators,
        block_size=config.block_size,
    )


class TritonVectorFmaOperatorTest(VectorFmaOperatorTestBase):
    """One cached raw Triton launch with an explicit preallocated output."""

    _compiled_probe_keys = set()

    def __init__(
        self,
        precision: PrecisionType,
        config: VectorFmaConfig,
    ) -> None:
        super().__init__(precision, config)
        self.provider_name = (
            FP16_PROVIDER
            if precision is PrecisionType.FP16
            else BF16_PROVIDER
        )

    def _require_runtime(self) -> None:
        error = _triton_capability_error()
        if error is not None:
            raise RuntimeError(error)

    def get_available_implementations(self, device: str) -> List[str]:
        if not (device.startswith("cuda") or device.startswith("npu")):
            return []
        if _triton_capability_error() is not None:
            return []
        return [self.provider_name]

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
        self._require_runtime()
        validate_triton_launch_geometry(self.config)
        prepared = super()._prepare_data_for_core_operator(
            data,
            device,
            precision,
            selected,
        )
        prepared["kernel"] = partial(
            _launch_vector_fma_once,
            vector_fma_kernel,
            prepared["a"],
            prepared["b"],
            prepared["output"],
            prepared["config"],
        )

        probe_key = (
            device,
            precision.name,
            prepared["config"],
        )
        if probe_key not in self._compiled_probe_keys:
            initial_output = prepared["output"].clone()
            prepared["kernel"]()
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            elif device.startswith("npu"):
                import torch_npu

                torch_npu.npu.synchronize()
            prepared["output"].copy_(initial_output)
            self._compiled_probe_keys.add(probe_key)
        return prepared

    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> torch.Tensor:
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
