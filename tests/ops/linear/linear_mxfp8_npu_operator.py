"""Ascend 950PR group32 MXFP8 Linear provider."""

from typing import Any, Dict

import torch

from linear.linear_fp8_npu_common import LinearFp8NpuBaseOperatorTest
from operator_test_framework import PrecisionType


class LinearMxFp8NpuOperatorTest(LinearFp8NpuBaseOperatorTest):
    """E4M3 data with group32 pair-packed E8M0 scales on Ascend 950PR."""

    NPU_IMPLEMENTATION = (
        "npu_quant_matmul_mxfp8_e4m3_e8m0_group32_bf16"
    )
    PRECISION = PrecisionType.MXFP8
    GROUP_SIZE = 32
    REQUIRED_RUNTIME_SYMBOLS = (
        "npu_dynamic_mx_quant",
        "npu_quant_matmul",
    )
    REQUIRES_E8M0_SCALE_DTYPE = True

    def __init__(self):
        super().__init__("LinearMxFp8Npu")

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

    @classmethod
    def _e8m0_scale_dtype(cls):
        runtime = cls._load_torch_npu()
        dtype = getattr(
            runtime,
            "float8_e8m0fnu",
            getattr(torch, "float8_e8m0fnu", None),
        )
        if dtype is None:
            raise RuntimeError(
                "MXFP8 Linear requires the float8_e8m0fnu scale dtype"
            )
        return dtype

    @classmethod
    def _normalize_pair_packed_scale(
        cls,
        scale: torch.Tensor,
        rows: int,
        width: int,
        label: str,
    ) -> torch.Tensor:
        group_count = width // cls.GROUP_SIZE
        expected_shape = (rows, group_count // 2, 2)
        if scale.ndim == 2 and tuple(scale.shape) == (rows, group_count):
            scale = scale.reshape(expected_shape)
        if tuple(scale.shape) != expected_shape:
            raise RuntimeError(
                f"{label} MXFP8 scale must have pair-packed shape "
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
        if input_dim % 64 != 0:
            raise ValueError(
                "Ascend MXFP8 Linear requires K to be a multiple of 64 "
                "so two group32 E8M0 scales form complete storage pairs; "
                f"got K={input_dim}"
            )
        if output_dim % 16 != 0:
            raise ValueError(
                "Ascend MXFP8 quant matmul requires N to be a multiple of "
                f"16; got N={output_dim}"
            )

        dynamic_mx_quant = self._dynamic_mx_quant_callable()
        activation_fp8, activation_scale = dynamic_mx_quant(
            input_source,
            dst_type=torch.float8_e4m3fn,
            block_size=self.GROUP_SIZE,
            round_mode="rint",
        )
        weight_fp8_nk, weight_scale_nk = dynamic_mx_quant(
            weight_source,
            dst_type=torch.float8_e4m3fn,
            block_size=self.GROUP_SIZE,
            round_mode="rint",
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
            "A": activation_fp8,
            "B": weight_fp8_nk.transpose(0, 1).contiguous(),
            "scale_a": activation_scale,
            "scale_b": weight_scale_nk.transpose(0, 1).contiguous(),
            "scale_dtype": self._e8m0_scale_dtype(),
            "group_size": self.GROUP_SIZE,
            "provider_metadata": {
                "weight_dtype": "float8_e4m3fn",
                "activation_dtype": "float8_e4m3fn",
                "weight_scale": "group32_float8_e8m0fnu",
                "activation_scale": "group32_float8_e8m0fnu",
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
                f"unsupported MXFP8 Linear implementation: {impl}"
            )
        scale_dtype = prepared_data["scale_dtype"]
        group_size = prepared_data["group_size"]
        return prepared_data["op"](
            prepared_data["A"],
            prepared_data["B"],
            prepared_data["scale_b"],
            pertoken_scale=prepared_data["scale_a"],
            bias=None,
            output_dtype=torch.bfloat16,
            scale_dtype=scale_dtype,
            pertoken_scale_dtype=scale_dtype,
            group_sizes=[1, 1, group_size],
        )
