"""Shared contract for independent Ascend 950PR FP8 Linear providers."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from operator_test_framework import BaseOperatorTest, DeviceType, PrecisionType


class LinearFp8NpuBaseOperatorTest(BaseOperatorTest):
    """Common Framework V2 plumbing; subclasses own one native provider."""

    NPU_IMPLEMENTATION = ""
    PRECISION = PrecisionType.FP8
    REQUIRED_RUNTIME_SYMBOLS = ("npu_quant_matmul",)
    REQUIRES_E8M0_SCALE_DTYPE = False

    def __init__(self, operator_name: str):
        super().__init__(operator_name)
        self.supported_precisions = [self.PRECISION]
        self.supported_devices = [DeviceType.NPU]

    def generate_test_data(
        self,
        batch_size: int = 32,
        input_dim: int = 1024,
        output_dim: int = 512,
        bias: bool = False,
        value_range: tuple = (-1.0, 1.0),
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate one bias-free source payload before native quantization."""
        del kwargs
        if bias:
            raise ValueError("Ascend FP8 Linear providers do not support bias")
        if min(batch_size, input_dim, output_dim) <= 0:
            raise ValueError("Linear dimensions must be positive")
        low, high = value_range
        input_tensor = (
            torch.rand(batch_size, input_dim) * (high - low) + low
        )
        weight_tensor = (
            torch.rand(output_dim, input_dim) * (high - low) + low
        )
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
                "operator_type": self.operator_name,
                "total_elements": (
                    input_tensor.numel() + weight_tensor.numel()
                ),
                "flops": batch_size * input_dim * output_dim * 2,
            },
        }

    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
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

    @staticmethod
    def _load_torch_npu():
        try:
            import torch_npu
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "Ascend FP8 Linear requires a working torch_npu runtime"
            ) from error
        return torch_npu

    def _npu_device_name(self, device: str) -> Optional[str]:
        if not device.startswith("npu"):
            return None
        try:
            index = int(device.split(":", maxsplit=1)[1]) if ":" in device else 0
            return str(self._load_torch_npu().npu.get_device_name(index))
        except (
            AttributeError,
            IndexError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return None

    def _runtime_supported(self) -> bool:
        try:
            runtime = self._load_torch_npu()
        except RuntimeError:
            return False
        if any(
            not callable(getattr(runtime, symbol, None))
            for symbol in self.REQUIRED_RUNTIME_SYMBOLS
        ):
            return False
        if self.REQUIRES_E8M0_SCALE_DTYPE and (
            getattr(runtime, "float8_e8m0fnu", None) is None
            and getattr(torch, "float8_e8m0fnu", None) is None
        ):
            return False
        return True

    def get_formal_implementations(self, device: str) -> List[str]:
        if not device.startswith("npu"):
            return []
        device_name = self._npu_device_name(device)
        if (
            device_name is not None
            and device_name.upper().startswith("ASCEND950PR")
            and self._runtime_supported()
        ):
            return [self.NPU_IMPLEMENTATION]
        return []

    def get_available_implementations(self, device: str) -> List[str]:
        return self.get_formal_implementations(device)

    def _resolve_implementation(
        self,
        device: str,
        implementation: str,
    ) -> str:
        formal = self.get_formal_implementations(device)
        if implementation == "default":
            if not formal:
                raise ValueError(
                    f"{self.operator_name} requires an Ascend950PR device; "
                    f"unsupported device {device}"
                )
            return formal[0]
        if implementation not in formal:
            raise ValueError(
                f"implementation {implementation!r} requires Ascend950PR; "
                f"device={device}, formal={formal}"
            )
        return implementation

    def _validate_and_copy_sources(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str,
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        resolved = self._resolve_implementation(device, implementation)
        if precision is not self.PRECISION:
            raise ValueError(
                f"{self.operator_name} requires "
                f"PrecisionType.{self.PRECISION.name}"
            )
        if data.get("bias") is not None and data.get("bias") is not False:
            raise ValueError("Ascend FP8 Linear providers do not support bias")

        input_tensor = data["input"]
        weight = data["weight"]
        if input_tensor.ndim != 2 or weight.ndim != 2:
            raise ValueError("Ascend FP8 Linear requires 2D input and weight")
        if input_tensor.shape[1] != weight.shape[1]:
            raise ValueError(
                "Linear input and weight K dimensions must match; "
                f"got {input_tensor.shape[1]} and {weight.shape[1]}"
            )
        return (
            resolved,
            input_tensor.to(
                device=device,
                dtype=torch.bfloat16,
                copy=True,
            ),
            weight.to(
                device=device,
                dtype=torch.bfloat16,
                copy=True,
            ),
        )

    @classmethod
    def _quant_matmul_callable(cls):
        operator = getattr(cls._load_torch_npu(), "npu_quant_matmul", None)
        if not callable(operator):
            raise RuntimeError("torch_npu.npu_quant_matmul is unavailable")
        return operator

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
