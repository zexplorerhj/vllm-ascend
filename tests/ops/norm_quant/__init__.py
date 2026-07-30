"""Fused RMSNorm and activation-quantization benchmark providers."""

from .base import NormQuantOperatorTestBase, NormQuantVariant
from .cuda_impl import CudaNormQuantOperatorTest
from operator_test_framework import BaseOperatorTest, PrecisionType


def create_norm_quant_operator(
    device: str,
    variant: NormQuantVariant,
    precision: PrecisionType,
) -> BaseOperatorTest:
    """Create only a native provider for the requested platform contract."""
    if device.startswith("cuda"):
        return CudaNormQuantOperatorTest(variant, precision)
    raise ValueError(
        f"NormQuant does not support device {device!r} for "
        f"variant={variant.value}, precision={precision.name}"
    )


__all__ = [
    "CudaNormQuantOperatorTest",
    "NormQuantOperatorTestBase",
    "NormQuantVariant",
    "create_norm_quant_operator",
]
