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
DEFAULT_TRITON_VECTOR_FMA_CONFIG = VectorFmaConfig(
    elements=4096,
    fma_depth=1024,
    accumulators=8,
    block_size=256,
    num_programs=16,
)


if triton is not None:

    @triton.jit
    def vector_fma_kernel(
        a_ptr,
        b_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        fma_depth: tl.constexpr,
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
    if config.block_size & (config.block_size - 1):
        raise ValueError(
            "block_size must be a power of two for tl.arange, got "
            f"{config.block_size}"
        )
    capacity = config.block_size * config.num_programs
    if capacity != config.elements:
        raise ValueError(
            "block_size * num_programs must be exactly equal to the global "
            "scalar lanes so physical and accounted FMA lanes match: "
            f"{capacity} != {config.elements}"
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
    _unsupported_reasons: Dict[tuple, str] = {}

    def __init__(
        self,
        precision: PrecisionType,
        config: VectorFmaConfig = DEFAULT_TRITON_VECTOR_FMA_CONFIG,
    ) -> None:
        super().__init__(precision, config)
        self.provider_name = (
            FP16_PROVIDER
            if precision is PrecisionType.FP16
            else BF16_PROVIDER
        )

    def _unsupported_key(self, device: str) -> tuple:
        return (device, self.provider_name)

    def _require_runtime(self, device: Optional[str] = None) -> None:
        error = _triton_capability_error()
        if error is None and device is not None:
            error = self._unsupported_reasons.get(
                self._unsupported_key(device)
            )
        if error is not None:
            raise RuntimeError(error)

    def get_available_implementations(self, device: str) -> List[str]:
        if not (device.startswith("cuda") or device.startswith("npu")):
            return []
        if _triton_capability_error() is not None:
            return []
        if self._unsupported_key(device) in self._unsupported_reasons:
            return []
        return [self.provider_name]

    def _probe_compilation(
        self,
        prepared: Dict[str, Any],
        device: str,
        precision: PrecisionType,
    ) -> None:
        probe_key = (device, precision.name, prepared["config"])
        if probe_key in self._compiled_probe_keys:
            return

        output = prepared.get("output")
        initial_output = (
            output.clone()
            if isinstance(output, torch.Tensor)
            else None
        )
        try:
            prepared["kernel"]()
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            elif device.startswith("npu"):
                import torch_npu

                torch_npu.npu.synchronize()
        except Exception as exc:
            reason = (
                f"{self.provider_name} unsupported on {device}: "
                f"{type(exc).__name__}: {exc}"
            )
            self._unsupported_reasons[
                self._unsupported_key(device)
            ] = reason
            raise RuntimeError(reason) from exc

        if initial_output is not None:
            output.copy_(initial_output)
        self._compiled_probe_keys.add(probe_key)

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
        self._require_runtime(device)
        validate_triton_launch_geometry(self._data_config(data))
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
        self._probe_compilation(prepared, device, precision)
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
