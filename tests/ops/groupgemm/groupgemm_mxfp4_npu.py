"""Ascend 950PR MXFP4 E2M1/E8M0 pure GroupGemm provider."""

from typing import Any, Dict, Tuple

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from groupgemm.groupgemm_fp8_npu import (
    _BaseGroupGemmFp8NpuOperatorTest,
)
from operator_test_framework import PrecisionType


class GroupGemmMxFp4NpuOperatorTest(
    _BaseGroupGemmFp8NpuOperatorTest
):
    """950PR pair-packed group-32 MXFP4 pure GMM2."""

    NPU_MXFP4_IMPLEMENTATION = (
        "npu_grouped_matmul_mxfp4_e2m1_e8m0_group32_bf16"
    )
    IMPLEMENTATION = NPU_MXFP4_IMPLEMENTATION
    QUANTIZATION_API = "npu_dynamic_mx_quant"
    PROVIDER_LABEL = "950PR MXFP4 GroupGemm"
    K_ALIGNMENT = 64
    KERNEL = "npu_grouped_matmul_mxfp4_gmm2"
    GROUP_SIZE = 32
    PRECISION = PrecisionType.MXFP4
    MXFP4_DTYPE_CODE = 296
    E8M0_DTYPE_CODE = 293

    def __init__(
        self,
        num_experts: int = 8,
        hidden_dim: int = 7168,
        out_channel: int = 4096,
        use_nz_format: bool = False,
    ):
        super().__init__(
            operator_name="GroupGemm_MXFP4",
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            out_channel=out_channel,
            use_nz_format=use_nz_format,
        )

    @staticmethod
    def _runtime_module():
        return torch_npu

    def _runtime_supported(self) -> bool:
        runtime = self._runtime_module()
        return bool(
            super()._runtime_supported()
            and getattr(runtime, "float4_e2m1fn_x2", None) is not None
            and getattr(runtime, "float8_e8m0fnu", None) is not None
        )

    def _quantization_kwargs(self) -> Dict[str, Any]:
        return {
            "dst_type": self.MXFP4_DTYPE_CODE,
            "block_size": self.GROUP_SIZE,
            "round_mode": "round",
        }

    def _expected_quantized_shapes(
        self,
        *,
        seq_len: int,
        num_experts: int,
        hidden_dim: int,
        out_channel: int,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (
            (seq_len, hidden_dim // 2),
            (num_experts, out_channel, hidden_dim // 2),
        )

    def _quantized_storage_dtype(self) -> torch.dtype:
        return torch.uint8

    def _quantized_element_size(self) -> float:
        return 0.5

    def _materialize_weight(self, weight_enk: torch.Tensor) -> torch.Tensor:
        return weight_enk.transpose(1, 2)

    @classmethod
    def _validate_pair_packed_scale(
        cls,
        scale: torch.Tensor,
        *,
        prefix: Tuple[int, ...],
        hidden_dim: int,
        label: str,
    ) -> torch.Tensor:
        expected_shape = (*prefix, hidden_dim // cls.K_ALIGNMENT, 2)
        if (
            tuple(scale.shape) != expected_shape
            or scale.dtype != torch.uint8
        ):
            raise RuntimeError(
                f"npu_dynamic_mx_quant {label} MXFP4 scale must have "
                f"pair-packed uint8 shape {expected_shape}, got "
                f"{scale.dtype} {tuple(scale.shape)}"
            )
        return scale

    def _normalize_scales(
        self,
        *,
        activation_scale: torch.Tensor,
        weight_scale_en: torch.Tensor,
        seq_len: int,
        num_experts: int,
        hidden_dim: int,
        out_channel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        activation_scale = self._validate_pair_packed_scale(
            activation_scale,
            prefix=(seq_len,),
            hidden_dim=hidden_dim,
            label="activation",
        )
        weight_scale_en = self._validate_pair_packed_scale(
            weight_scale_en,
            prefix=(num_experts, out_channel),
            hidden_dim=hidden_dim,
            label="weight",
        )
        return activation_scale, weight_scale_en.transpose(1, 2)

    def _extra_grouped_matmul_kwargs(self) -> Dict[str, Any]:
        return {
            "x_dtype": self.MXFP4_DTYPE_CODE,
            "weight_dtype": self.MXFP4_DTYPE_CODE,
            "scale_dtype": self.E8M0_DTYPE_CODE,
            "per_token_scale_dtype": self.E8M0_DTYPE_CODE,
        }

    def _scale_storage_dtype(self) -> torch.dtype:
        return torch.uint8

    def _scale_payload_bytes(
        self,
        *,
        seq_len: int,
        num_experts: int,
        hidden_dim: int,
        out_channel: int,
    ) -> int:
        activation_scale_bytes = (
            seq_len * hidden_dim // self.GROUP_SIZE
        )
        weight_scale_bytes = (
            num_experts
            * out_channel
            * hidden_dim
            // self.GROUP_SIZE
        )
        return activation_scale_bytes + weight_scale_bytes
