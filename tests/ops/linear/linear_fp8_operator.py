"""H20 FP8 W8A8 Linear provider backed by vLLM CUTLASS."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from fp8_utils import (
    quantize_fp8_per_row,
    quantize_fp8_weight_per_channel,
    resolve_vllm_cutlass_scaled_mm,
)
from operator_test_framework import BaseOperatorTest, DeviceType, PrecisionType


class LinearFp8OperatorTest(BaseOperatorTest):
    """Preallocated-output FP8 Linear benchmark provider for CUDA."""

    CUDA_IMPLEMENTATION = "cuda_vllm_cutlass_scaled_mm_fp8_bf16"

    def __init__(self):
        super().__init__("LinearFp8")
        self.supported_precisions = [PrecisionType.FP8]
        self.supported_devices = [DeviceType.GPU]

    def generate_test_data(
        self,
        batch_size: int = 32,
        input_dim: int = 1024,
        output_dim: int = 512,
        bias: bool = False,
        value_range: tuple = (-1.0, 1.0),
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate source tensors for a bias-free FP8 GEMM payload."""
        if bias:
            raise ValueError("FP8 Linear CUTLASS provider does not support bias")
        low, high = value_range
        input_tensor = torch.rand(batch_size, input_dim) * (high - low) + low
        weight_tensor = torch.rand(output_dim, input_dim) * (high - low) + low
        return {
            "input": input_tensor,
            "weight": weight_tensor,
            "bias": False,
            "metadata": {
                "batch_size": batch_size,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "has_bias": False,
                "value_range": value_range,
                "operator_type": "LinearFp8",
                "total_elements": input_tensor.numel() + weight_tensor.numel(),
                "flops": batch_size * input_dim * output_dim * 2,
            },
        }

    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """Return the unquantized bias-free reference result."""
        return F.linear(data["input"].cpu(), data["weight"].cpu(), None)

    def run_device_implementation(
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

    def run_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> torch.Tensor:
        return self.run_device_implementation(
            data,
            device,
            precision,
            implementation,
        )

    def _prepare_data_for_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Dict[str, Any]:
        """Quantize a fresh payload outside the timed CUTLASS invocation."""
        implementation = self._resolve_implementation(device, implementation)
        if precision is not PrecisionType.FP8:
            raise ValueError("FP8 Linear provider requires PrecisionType.FP8")
        if data.get("bias") is not None and data.get("bias") is not False:
            raise ValueError("FP8 Linear CUTLASS provider does not support bias")

        input_dim = data["input"].shape[1]
        output_dim = data["weight"].shape[0]
        if input_dim % 16 != 0 or output_dim % 16 != 0:
            raise ValueError(
                "FP8 CUTLASS scaled-mm requires K and N to be multiples "
                f"of 16; got K={input_dim}, N={output_dim}"
            )
        cutlass_scaled_mm = self._cutlass_scaled_mm()

        input_source = data["input"].to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )
        weight_source = data["weight"].to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )
        activation_fp8, activation_scale = quantize_fp8_per_row(input_source)
        weight_fp8_nk, weight_scale_n1 = quantize_fp8_weight_per_channel(
            weight_source
        )
        return {
            "implementation": implementation,
            "op": cutlass_scaled_mm,
            "A": activation_fp8,
            "B": weight_fp8_nk.t(),
            "scale_a": activation_scale,
            "scale_b": weight_scale_n1.t().contiguous(),
            "output": torch.empty(
                input_source.shape[0],
                weight_source.shape[0],
                dtype=torch.bfloat16,
                device=device,
            ),
        }

    @staticmethod
    def _cutlass_scaled_mm():
        return resolve_vllm_cutlass_scaled_mm()

    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> torch.Tensor:
        """Execute only the caller-output CUTLASS scaled-mm kernel."""
        del implementation
        prepared_data["op"](
            prepared_data["output"],
            prepared_data["A"],
            prepared_data["B"],
            prepared_data["scale_a"],
            prepared_data["scale_b"],
            None,
        )
        return prepared_data["output"]

    def _declares_preallocated_output_contract(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> bool:
        return (
            prepared_data.get("implementation", implementation)
            == self.CUDA_IMPLEMENTATION
        )

    def get_available_implementations(self, device: str) -> List[str]:
        return self.get_formal_implementations(device)

    @staticmethod
    def _cuda_device_capability(device: str) -> Optional[tuple[int, int]]:
        if not device.startswith("cuda"):
            return None
        try:
            return tuple(torch.cuda.get_device_capability(device))
        except (AssertionError, RuntimeError, TypeError, ValueError):
            return None

    def get_formal_implementations(self, device: str) -> List[str]:
        if self._cuda_device_capability(device) == (9, 0):
            return [self.CUDA_IMPLEMENTATION]
        return []

    def _resolve_implementation(self, device: str, implementation: str) -> str:
        formal = self.get_formal_implementations(device)
        if implementation == "default":
            if not formal:
                raise ValueError(
                    "FP8 Linear requires an exact CUDA SM90/H20 device; "
                    f"unsupported device {device}"
                )
            return formal[0]
        if implementation not in formal:
            raise ValueError(
                f"implementation {implementation!r} requires exact CUDA "
                f"SM90/H20; device={device}, formal={formal}"
            )
        return implementation

    def calculate_flops(self, data: Dict[str, Any]) -> Optional[float]:
        return data.get("metadata", {}).get("flops")

    def calculate_throughput(
        self,
        data: Dict[str, Any],
        time_ms: float,
    ) -> float:
        flops = self.calculate_flops(data)
        if flops is None or time_ms <= 0:
            return 0.0
        return (flops / (time_ms / 1000.0)) / 1e9
