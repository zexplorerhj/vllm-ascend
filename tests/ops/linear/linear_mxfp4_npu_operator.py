"""Ascend 950PR native group32 MXFP4 Linear provider."""

from typing import Any, Dict

import torch

from linear.linear_fp8_npu_common import LinearFp8NpuBaseOperatorTest
from operator_test_framework import PrecisionType


class LinearMxFp4NpuOperatorTest(LinearFp8NpuBaseOperatorTest):
    """Pair-packed E2M1 data with pair-packed E8M0 group scales."""

    NPU_IMPLEMENTATION = (
        "npu_quant_matmul_mxfp4_e2m1_e8m0_group32_bf16"
    )
    PRECISION = PrecisionType.MXFP4
    GROUP_SIZE = 32
    K_ALIGNMENT = 64
    MXFP4_DTYPE_CODE = 296
    E8M0_DTYPE_CODE = 293
    REQUIRED_RUNTIME_SYMBOLS = (
        "npu_dynamic_mx_quant",
        "npu_quant_matmul",
    )

    def __init__(self):
        super().__init__("LinearMxFp4Npu")

    @classmethod
    def _dynamic_mx_quant_callable(cls):
        operator = getattr(
            cls._load_torch_npu(),
            "npu_dynamic_mx_quant",
            None,
        )
        if not callable(operator):
            raise RuntimeError(
                "torch_npu.npu_dynamic_mx_quant is unavailable"
            )
        return operator

    def _runtime_supported(self) -> bool:
        if not super()._runtime_supported():
            return False
        try:
            runtime = self._load_torch_npu()
        except RuntimeError:
            return False
        return (
            getattr(runtime, "float4_e2m1fn_x2", None) is not None
            and getattr(runtime, "float8_e8m0fnu", None) is not None
        )

    @classmethod
    def _normalize_pair_packed_scale(
        cls,
        scale: torch.Tensor,
        rows: int,
        width: int,
        label: str,
    ) -> torch.Tensor:
        expected_shape = (rows, width // cls.K_ALIGNMENT, 2)
        if tuple(scale.shape) != expected_shape:
            raise RuntimeError(
                f"{label} MXFP4 scale must have pair-packed shape "
                f"{expected_shape}; got {tuple(scale.shape)}"
            )
        return scale.contiguous()

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
        if input_dim % self.K_ALIGNMENT != 0:
            raise ValueError(
                "Ascend MXFP4 Linear requires K to be a multiple of "
                f"{self.K_ALIGNMENT}; got K={input_dim}"
            )
        if output_dim % 16 != 0:
            raise ValueError(
                "Ascend MXFP4 quant matmul requires N to be a multiple of "
                f"16; got N={output_dim}"
            )

        dynamic_mx_quant = self._dynamic_mx_quant_callable()
        activation_packed, activation_scale = dynamic_mx_quant(
            input_source,
            dst_type=self.MXFP4_DTYPE_CODE,
            block_size=self.GROUP_SIZE,
            round_mode="round",
        )
        weight_packed_nk, weight_scale_nk = dynamic_mx_quant(
            weight_source,
            dst_type=self.MXFP4_DTYPE_CODE,
            block_size=self.GROUP_SIZE,
            round_mode="round",
        )
        activation_scale = self._normalize_pair_packed_scale(
            activation_scale,
            input_source.shape[0],
            input_dim,
            "activation",
        )
        weight_scale_nk = self._normalize_pair_packed_scale(
            weight_scale_nk,
            output_dim,
            input_dim,
            "weight",
        )
        return {
            "implementation": implementation,
            "op": self._quant_matmul_callable(),
            "A": activation_packed,
            "B": weight_packed_nk.transpose(0, 1),
            "scale_a": activation_scale,
            "scale_b": weight_scale_nk.transpose(0, 1),
            "x_dtype": self.MXFP4_DTYPE_CODE,
            "scale_dtype": self.E8M0_DTYPE_CODE,
        }

    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> torch.Tensor:
        impl = prepared_data.get("implementation", implementation)
        if impl != self.NPU_IMPLEMENTATION:
            raise ValueError(
                f"unsupported MXFP4 Linear implementation: {impl}"
            )
        x_dtype = prepared_data["x_dtype"]
        scale_dtype = prepared_data["scale_dtype"]
        return prepared_data["op"](
            prepared_data["A"],
            prepared_data["B"],
            prepared_data["scale_b"],
            pertoken_scale=prepared_data["scale_a"],
            output_dtype=torch.bfloat16,
            x1_dtype=x_dtype,
            x2_dtype=x_dtype,
            scale_dtype=scale_dtype,
            pertoken_scale_dtype=scale_dtype,
            group_sizes=[1, 1, self.GROUP_SIZE],
        )
