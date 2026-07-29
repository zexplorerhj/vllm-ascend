"""Ascend 950PR plain W8A8 E4M3 Linear provider."""

from typing import Any, Dict

import torch

from linear.linear_fp8_npu_common import LinearFp8NpuBaseOperatorTest
from operator_test_framework import PrecisionType


class LinearFp8NpuOperatorTest(LinearFp8NpuBaseOperatorTest):
    """Per-token/per-channel FP32-scale FP8 Linear on Ascend 950PR."""

    NPU_IMPLEMENTATION = (
        "npu_quant_matmul_fp8_e4m3_per_token_per_channel_bf16"
    )
    REQUIRED_RUNTIME_SYMBOLS = (
        "npu_dynamic_quant",
        "npu_quant_matmul",
    )

    def __init__(self):
        super().__init__("LinearFp8Npu")

    @classmethod
    def _dynamic_quant_callable(cls):
        operator = getattr(cls._load_torch_npu(), "npu_dynamic_quant", None)
        if not callable(operator):
            raise RuntimeError("torch_npu.npu_dynamic_quant is unavailable")
        return operator

    def _prepare_data_for_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Dict[str, Any]:
        implementation, input_source, weight_source = (
            self._validate_and_copy_sources(
                data,
                device,
                precision,
                implementation,
            )
        )
        input_dim = input_source.shape[1]
        output_dim = weight_source.shape[0]
        if input_dim % 16 != 0 or output_dim % 16 != 0:
            raise ValueError(
                "Ascend plain FP8 quant matmul requires K and N to be "
                f"multiples of 16; got K={input_dim}, N={output_dim}"
            )

        dynamic_quant = self._dynamic_quant_callable()
        activation_fp8, activation_scale = dynamic_quant(
            input_source,
            dst_type=torch.float8_e4m3fn,
        )
        weight_fp8_nk, weight_scale = dynamic_quant(
            weight_source,
            dst_type=torch.float8_e4m3fn,
        )
        activation_scale = activation_scale.reshape(-1).contiguous()
        weight_scale = weight_scale.reshape(-1).contiguous()
        if activation_scale.numel() != input_source.shape[0]:
            raise RuntimeError(
                "plain FP8 activation scale must contain one value per token"
            )
        if weight_scale.numel() != weight_source.shape[0]:
            raise RuntimeError(
                "plain FP8 weight scale must contain one value per output "
                "channel"
            )
        return {
            "implementation": implementation,
            "op": self._quant_matmul_callable(),
            "A": activation_fp8,
            "B": weight_fp8_nk.transpose(0, 1).contiguous(),
            "scale_a": activation_scale,
            "scale_b": weight_scale,
            "provider_metadata": {
                "weight_dtype": "float8_e4m3fn",
                "activation_dtype": "float8_e4m3fn",
                "weight_scale": "per_output_channel_fp32",
                "activation_scale": "per_token_fp32",
                "output_dtype": "bfloat16",
            },
        }

    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> torch.Tensor:
        impl = prepared_data.get("implementation", implementation)
        if impl != self.NPU_IMPLEMENTATION:
            raise ValueError(
                f"unsupported plain FP8 Linear implementation: {impl}"
            )
        return prepared_data["op"](
            prepared_data["A"],
            prepared_data["B"],
            prepared_data["scale_b"],
            pertoken_scale=prepared_data["scale_a"],
            bias=None,
            output_dtype=torch.bfloat16,
        )
