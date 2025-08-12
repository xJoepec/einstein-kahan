"""
Core Kahan summation implementations.

This module contains the fundamental classes for Kahan compensated summation
and Einstein-Kahan geometric error correction.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Union
import numpy as np


class KahanAccumulator:
    """
    Kahan summation accumulator for numerical stability.
    
    Implements the classic Kahan compensated summation algorithm to minimize
    the accumulation of floating-point rounding errors.
    
    Attributes:
        sum: The accumulated sum
        c: The compensation term tracking lost precision
    """
    
    def __init__(self, shape=(), dtype=torch.float32, device=None):
        """
        Initialize Kahan accumulator.
        
        Args:
            shape: Shape of the accumulator tensor
            dtype: Data type for the accumulator
            device: Device to place the tensors on
        """
        self.sum = torch.zeros(shape, dtype=dtype, device=device)
        self.c = torch.zeros(shape, dtype=dtype, device=device)  # Compensation
        self.dtype = dtype
        self.device = device or torch.device('cpu')
    
    def add(self, value: Union[torch.Tensor, float]):
        """
        Add value with Kahan compensation.
        
        Args:
            value: Value to add to the accumulator
        """
        if isinstance(value, (int, float)):
            value = torch.tensor(value, dtype=self.dtype, device=self.device)
        
        y = value - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
    
    def get(self) -> torch.Tensor:
        """Get compensated sum."""
        return self.sum
    
    def reset(self):
        """Reset the accumulator to zero."""
        self.sum.zero_()
        self.c.zero_()


class EinsteinKahanSummation(nn.Module):
    """
    Einstein-Kahan summation for geometric operations.
    
    Combines Einstein summation convention with Kahan error compensation
    and geometric curvature correction inspired by the 8π factor in
    Einstein's field equations.
    """
    
    def __init__(self, dim: int, einstein_constant: Optional[float] = None):
        """
        Initialize Einstein-Kahan summation module.
        
        Args:
            dim: Dimension of the tensors being operated on
            einstein_constant: Geometric normalization constant (default: 8π)
        """
        super().__init__()
        self.dim = dim
        self.einstein_constant = einstein_constant or (8 * math.pi)
        
        # Track numerical errors in higher precision
        self.error_accumulator = KahanAccumulator((dim,), dtype=torch.float64)
        
        # Register as buffer for state preservation
        self.register_buffer('total_operations', torch.tensor(0, dtype=torch.long))
    
    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor, 
                equation: str = "ij,jk->ik") -> torch.Tensor:
        """
        Perform Einstein summation with Kahan compensation.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor  
            equation: Einstein summation equation
            
        Returns:
            Result with geometric error compensation
        """
        # Standard einsum operation
        result = torch.einsum(equation, tensor1, tensor2)
        
        # Track error in higher precision for curvature analysis
        result_fp64 = torch.einsum(equation, tensor1.double(), tensor2.double())
        error = (result_fp64 - result.double()).float()
        
        # Apply Einstein geometric correction
        # The 8π factor normalizes local curvature with global structure
        correction = self.einstein_constant * error / self.dim
        result = result + correction
        
        # Update error tracking
        self.error_accumulator.add(error.mean())
        self.total_operations += 1
        
        return result
    
    def get_total_error(self) -> float:
        """Get accumulated numerical error."""
        return self.error_accumulator.get().item()
    
    def get_operation_count(self) -> int:
        """Get total number of operations performed."""
        return self.total_operations.item()
    
    def reset_error_tracking(self):
        """Reset error accumulation."""
        self.error_accumulator.reset()
        self.total_operations.zero_()


def kahan_add(a: float, b: float, c: float = 0.0) -> Tuple[float, float]:
    """
    Single-step Kahan addition.
    
    Args:
        a: First value
        b: Second value  
        c: Current compensation term
        
    Returns:
        Tuple of (new_sum, new_compensation)
    """
    y = b - c
    t = a + y
    new_c = (t - a) - y
    return t, new_c


def compensated_dot_product(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dot product with error compensation.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Tuple of (dot_product, accumulated_error)
    """
    assert a.shape == b.shape, "Vectors must have same shape"
    
    acc = KahanAccumulator(dtype=a.dtype, device=a.device)
    
    # Element-wise multiplication and compensated summation
    products = a * b
    for product in products.flatten():
        acc.add(product)
    
    return acc.get(), acc.c.sum()


class RiemannCurvatureTensor(nn.Module):
    """
    Riemann curvature tensor for geometric computations.
    
    Tracks the intrinsic curvature of the attention manifold in transformer
    operations, providing geometric insight into numerical error patterns.
    """
    
    def __init__(self, dim: int, device=None):
        """
        Initialize Riemann curvature tensor.
        
        Args:
            dim: Dimension of the manifold
            device: Device to place tensors on
        """
        super().__init__()
        self.dim = dim
        self.device = device or torch.device('cpu')
        
        # Store only essential components due to symmetries
        # R_ijkl = -R_jikl = -R_ijlk = R_klij
        max_dim = min(dim, 32)  # Cap to avoid memory explosion
        self.n_components = min((max_dim * (max_dim - 1) // 2) ** 2, 10000)
        
        # Compressed storage for curvature components
        self.components = nn.Parameter(
            torch.zeros(self.n_components, device=device)
        )
        
        # Build index mapping for symmetries
        self._build_index_map()
    
    def _build_index_map(self):
        """Build mapping from full indices to compressed storage."""
        self.index_map = {}
        idx = 0
        max_dim = min(self.dim, 32)
        
        for i in range(max_dim):
            for j in range(i+1, max_dim):
                for k in range(max_dim):
                    for l in range(k+1, max_dim):
                        if idx >= self.n_components:
                            return
                        self.index_map[(i, j, k, l)] = idx
                        # Apply Riemann symmetries
                        self.index_map[(j, i, k, l)] = idx  # with sign flip
                        self.index_map[(i, j, l, k)] = idx  # with sign flip  
                        self.index_map[(k, l, i, j)] = idx
                        idx += 1
    
    def get_component(self, i: int, j: int, k: int, l: int) -> torch.Tensor:
        """Get R_ijkl with proper symmetries."""
        sign = 1.0
        
        # Anti-symmetry in first pair
        if i > j:
            i, j = j, i
            sign *= -1
        
        # Anti-symmetry in second pair  
        if k > l:
            k, l = l, k
            sign *= -1
        
        # Swap pairs if needed
        if (i, j) > (k, l):
            i, j, k, l = k, l, i, j
        
        # Get from compressed storage
        key = (i, j, k, l)
        if key in self.index_map:
            return sign * self.components[self.index_map[key]]
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_ricci(self) -> torch.Tensor:
        """Compute Ricci tensor by contraction: R_ij = R^k_ikj."""
        ricci = torch.zeros(self.dim, self.dim, device=self.device)
        
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    ricci[i, j] += self.get_component(k, i, k, j)
        
        return ricci
    
    def compute_scalar_curvature(self) -> torch.Tensor:
        """Compute scalar curvature R = g^ij R_ij."""
        ricci = self.compute_ricci()
        return torch.trace(ricci)
    
    def forward(self, metric: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute curvature quantities.
        
        Args:
            metric: Optional metric tensor (default: identity)
            
        Returns:
            Tuple of (Ricci tensor, Scalar curvature)
        """
        ricci = self.compute_ricci()
        scalar = self.compute_scalar_curvature()
        
        return ricci, scalar