"""
Kahan Summation Library

A high-precision numerical summation library implementing Kahan's compensated summation
algorithm and Einstein-Kahan geometric error correction for floating-point arithmetic.

This library provides:
- Standard Kahan compensated summation
- Einstein-Kahan summation with geometric error correction
- Parallel reduction algorithms
- C++ SIMD optimizations
- Support for multiple floating-point precisions
"""

from .core import KahanAccumulator, EinsteinKahanSummation
from .algorithms import (
    kahan_sum,
    einstein_kahan_sum,
    parallel_kahan_sum,
    tree_reduce_kahan
)

__version__ = "1.0.0"
__author__ = "Kahan Summation Contributors"

__all__ = [
    "KahanAccumulator",
    "EinsteinKahanSummation", 
    "kahan_sum",
    "einstein_kahan_sum",
    "parallel_kahan_sum",
    "tree_reduce_kahan"
]