"""Platform-independent vector FMA benchmark contract."""

from .base import (
    VectorFmaConfig,
    VectorFmaOperatorTestBase,
    vector_fma_arithmetic_intensity,
    vector_fma_flops,
)

__all__ = [
    "VectorFmaConfig",
    "VectorFmaOperatorTestBase",
    "vector_fma_arithmetic_intensity",
    "vector_fma_flops",
]
