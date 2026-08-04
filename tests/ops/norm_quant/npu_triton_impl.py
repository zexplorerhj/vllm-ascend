"""Preallocated Triton AddRMSNorm static-FP8 provider for Ascend 950PR."""

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError, RuntimeError):
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _add_rms_norm_static_fp8_kernel(
        output_ptr,
        x_ptr,
        residual_ptr,
        weight_ptr,
        scale_ptr,
        rows,
        cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        programs = tl.num_programs(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < cols
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scale_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
        for row in range(pid, rows, programs):
            row_offsets = row * cols + offsets
            summed = (
                tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(
                    tl.float32
                )
                + tl.load(
                    residual_ptr + row_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
            ).to(tl.bfloat16)
            tl.store(residual_ptr + row_offsets, summed, mask=mask)
            summed_f32 = summed.to(tl.float32)
            inv_rms = 1.0 / tl.sqrt(
                tl.sum(summed_f32 * summed_f32, axis=0) / cols + eps
            )
            normalized = (
                summed_f32 * inv_rms * weight
            ).to(tl.bfloat16).to(tl.float32)
            quantized = tl.maximum(
                -448.0,
                tl.minimum(448.0, normalized * scale_inv),
            )
            tl.store(
                output_ptr + row_offsets,
                quantized.to(output_ptr.dtype.element_ty),
                mask=mask,
            )


def _require_tensor(
    value: Any,
    *,
    label: str,
    ndim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{label} must be {ndim}-D")
    if value.dtype is not dtype:
        raise ValueError(f"{label} must have dtype {dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{label} must be contiguous")
    return value


def _num_vectorcore(device: torch.device) -> int:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    properties = triton.runtime.driver.active.utils.get_device_properties(
        device.index or 0
    )
    count = int(properties["num_vectorcore"])
    if count <= 0:
        raise RuntimeError("Triton reported no vector cores")
    return count


def run_triton_add_rms_norm_static_fp8_quant_out(
    output: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Quantize one BF16 AddRMSNorm result into the provided E4M3 buffer."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    output = _require_tensor(
        output,
        label="output",
        ndim=2,
        dtype=torch.float8_e4m3fn,
    )
    x = _require_tensor(x, label="x", ndim=2, dtype=torch.bfloat16)
    residual = _require_tensor(
        residual,
        label="residual",
        ndim=2,
        dtype=torch.bfloat16,
    )
    weight = _require_tensor(
        weight,
        label="weight",
        ndim=1,
        dtype=torch.bfloat16,
    )
    scale = _require_tensor(
        scale,
        label="scale",
        ndim=1,
        dtype=torch.float32,
    )
    if output.shape != x.shape or residual.shape != x.shape:
        raise ValueError("output, x, and residual must have the same shape")
    rows, cols = x.shape
    if weight.shape != (cols,):
        raise ValueError("weight must have shape [hidden]")
    if scale.shape != (1,):
        raise ValueError("scale must have shape [1]")
    if cols > 32768:
        raise ValueError("hidden dimension must be <= 32768")
    if not x.device.type.startswith("npu"):
        raise ValueError("Triton AddRMSNorm requires an NPU tensor")
    if any(tensor.device != x.device for tensor in (output, residual, weight, scale)):
        raise ValueError("all Triton AddRMSNorm tensors must share one device")
    # Ascend Triton accepts an exact constexpr arange width.  Keeping the
    # logical hidden size avoids carrying masked lanes through the full
    # Add/RMSNorm/quant pipeline (for example, 7168 -> 8192 wastes 12.5%).
    block_size = cols
    programs = min(rows, _num_vectorcore(x.device))
    _add_rms_norm_static_fp8_kernel[(programs,)](
        output,
        x,
        residual,
        weight,
        scale,
        rows,
        cols=cols,
        eps=eps,
        BLOCK_SIZE=block_size,
    )
    return output
