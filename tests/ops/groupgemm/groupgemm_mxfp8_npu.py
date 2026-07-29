"""Ascend 950PR MXFP8 E4M3/E8M0 pure GroupGemm provider."""

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


class GroupGemmMxFp8NpuOperatorTest(
    _BaseGroupGemmFp8NpuOperatorTest
):
    """950PR group-32 MXFP8 pure GMM2 without fused SwiGLU."""

    NPU_MXFP8_IMPLEMENTATION = (
        "npu_grouped_matmul_mxfp8_e4m3_e8m0_group32_bf16"
    )
    IMPLEMENTATION = NPU_MXFP8_IMPLEMENTATION
    QUANTIZATION_API = "npu_dynamic_mx_quant"
    PROVIDER_LABEL = "950PR MXFP8 GroupGemm"
    K_ALIGNMENT = 64
    KERNEL = "npu_grouped_matmul_mxfp8_gmm2"
    GROUP_SIZE = 32
    PRECISION = PrecisionType.MXFP8

    def __init__(
        self,
        num_experts: int = 8,
        hidden_dim: int = 7168,
        out_channel: int = 4096,
        use_nz_format: bool = False,
    ):
        super().__init__(
            operator_name="GroupGemm_MXFP8",
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
            and getattr(runtime, "float8_e8m0fnu", None) is not None
        )

    def _scale_storage_dtype(self) -> torch.dtype:
        # torch_npu currently exposes E8M0 scales as packed uint8 storage.
        return torch.uint8

    @staticmethod
    def _normalize_packed_e8m0(
        scale: torch.Tensor,
        *,
        prefix: Tuple[int, ...],
        hidden_dim: int,
        label: str,
    ) -> torch.Tensor:
        packed_shape = (*prefix, hidden_dim // 64, 2)
        flat_shape = (*prefix, hidden_dim // 32)
        if tuple(scale.shape) == packed_shape:
            return scale
        if tuple(scale.shape) == flat_shape:
            return scale.reshape(packed_shape)
        raise RuntimeError(
            f"npu_dynamic_mx_quant {label} scale must be packed E8M0 "
            f"{packed_shape} or flat {flat_shape}, got "
            f"{tuple(scale.shape)}"
        )

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
        activation_scale = self._normalize_packed_e8m0(
            activation_scale,
            prefix=(seq_len,),
            hidden_dim=hidden_dim,
            label="activation",
        )
        weight_scale_en = self._normalize_packed_e8m0(
            weight_scale_en,
            prefix=(num_experts, out_channel),
            hidden_dim=hidden_dim,
            label="weight",
        )
        # Runtime weight is [E,K,N], so its group scale layout is
        # [E,K/64,N,2], matching the production A5 pure-GMM2 path.
        weight_scale = weight_scale_en.transpose(1, 2).contiguous()
        return activation_scale, weight_scale

    def _extra_grouped_matmul_kwargs(self) -> Dict[str, Any]:
        scale_dtype = self._runtime_module().float8_e8m0fnu
        return {
            "scale_dtype": scale_dtype,
            "per_token_scale_dtype": scale_dtype,
        }

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
