"""Shared FP8 quantization and vLLM CUTLASS runtime helpers."""

import torch


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


def _quantize_fp8_per_channel(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize every row of a 2D tensor with its own FP32 scale."""
    if tensor.ndim != 2:
        raise ValueError("FP8 quantization requires a 2D tensor")

    source = tensor.to(dtype=torch.float32)
    amax = source.abs().amax(dim=1, keepdim=True)
    scales = amax / FP8_MAX
    scales = torch.where(amax == 0, torch.ones_like(scales), scales)
    quantized = (source / scales).to(dtype=FP8_DTYPE)
    return quantized, scales


def quantize_fp8_per_row(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations with one E4M3 scale per input row."""
    return _quantize_fp8_per_channel(tensor)


def quantize_fp8_weight_per_channel(
    weight_nk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [N, K] weights with one E4M3 scale per output channel."""
    return _quantize_fp8_per_channel(weight_nk)


def _load_vllm_custom_ops():
    """Import vLLM custom ops only when a CUDA provider is requested."""
    try:
        from vllm import _custom_ops as vllm_ops
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "vLLM CUTLASS operators are unavailable because vLLM custom ops "
            "could not be imported"
        ) from error
    return vllm_ops


def resolve_vllm_cutlass_scaled_mm():
    """Return the low-level CUTLASS scaled-mm operator with ``out`` support."""
    _load_vllm_custom_ops()
    operator = getattr(
        getattr(torch.ops, "_C", None),
        "cutlass_scaled_mm",
        None,
    )
    if not callable(operator):
        raise RuntimeError(
            "vLLM CUTLASS cutlass_scaled_mm operator is unavailable"
        )
    return operator


def resolve_vllm_cutlass_grouped_mm():
    """Return vLLM's FP8 CUTLASS grouped-MoE operator."""
    operator = getattr(_load_vllm_custom_ops(), "cutlass_moe_mm", None)
    if not callable(operator):
        raise RuntimeError(
            "vLLM CUTLASS cutlass_moe_mm operator is unavailable"
        )
    return operator
