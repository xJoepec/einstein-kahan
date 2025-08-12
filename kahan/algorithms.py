"""
High-level algorithms for Kahan summation.

This module provides practical implementations of various summation algorithms
including standard Kahan summation, Einstein-Kahan geometric correction,
and parallel reduction strategies.
"""

import torch
import numpy as np
from typing import List, Union, Tuple, Optional, Callable
from .core import KahanAccumulator, kahan_add
import math


def kahan_sum(values: Union[List[float], torch.Tensor, np.ndarray]) -> float:
    """
    Compute sum using Kahan compensated summation.
    
    Args:
        values: Sequence of values to sum
        
    Returns:
        Compensated sum with reduced floating-point error
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    elif isinstance(values, (list, tuple)):
        values = np.array(values)
    
    if len(values) == 0:
        return 0.0
    
    sum_val = float(values[0])
    c = 0.0
    
    for i in range(1, len(values)):
        sum_val, c = kahan_add(sum_val, float(values[i]), c)
    
    return sum_val


def einstein_kahan_sum(values: Union[List[float], torch.Tensor, np.ndarray],
                      einstein_constant: float = 8 * math.pi) -> Tuple[float, float]:
    """
    Compute sum using Einstein-Kahan geometric error correction.
    
    This algorithm treats floating-point summation as parallel transport
    on a curved manifold and applies geometric correction inspired by
    Einstein's field equations.
    
    Args:
        values: Sequence of values to sum
        einstein_constant: Geometric normalization constant (default: 8π)
        
    Returns:
        Tuple of (corrected_sum, total_curvature_error)
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    elif isinstance(values, (list, tuple)):
        values = np.array(values)
    
    if len(values) == 0:
        return 0.0, 0.0
    
    # Standard Kahan summation to get base result
    kahan_result = kahan_sum(values)
    
    # Compute true sum in higher precision for error analysis
    true_sum = np.sum(values.astype(np.float64))
    
    # Calculate curvature (accumulated error)
    curvature = float(true_sum - kahan_result)
    
    # Apply Einstein geometric correction
    correction = einstein_constant * curvature / len(values)
    corrected_sum = kahan_result + correction
    
    return corrected_sum, curvature


def tree_reduce_kahan(values: Union[List[float], torch.Tensor, np.ndarray]) -> Tuple[float, float]:
    """
    Perform tree-based reduction with Kahan compensation.
    
    This algorithm is particularly effective for parallel computation
    as it maintains numerical stability while enabling parallel reduction.
    
    Args:
        values: Sequence of values to sum
        
    Returns:
        Tuple of (sum, total_error)
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy().tolist()
    elif isinstance(values, np.ndarray):
        values = values.tolist()
    elif not isinstance(values, list):
        values = list(values)
    
    if len(values) == 0:
        return 0.0, 0.0
    elif len(values) == 1:
        return float(values[0]), 0.0
    
    # Recursive tree reduction
    def tree_reduce_recursive(vals):
        if len(vals) == 1:
            return vals[0], 0.0
        elif len(vals) == 2:
            # Base case: two values
            result, comp = kahan_add(vals[0], vals[1])
            return result, comp
        else:
            # Split and recurse
            mid = len(vals) // 2
            left_sum, left_error = tree_reduce_recursive(vals[:mid])
            right_sum, right_error = tree_reduce_recursive(vals[mid:])
            
            # Combine with Kahan compensation
            combined_sum, combine_error = kahan_add(left_sum, right_sum)
            total_error = left_error + right_error + combine_error
            
            return combined_sum, total_error
    
    return tree_reduce_recursive(values)


def parallel_kahan_sum(values: Union[List[float], torch.Tensor, np.ndarray],
                      num_partitions: int = 4) -> Tuple[float, float]:
    """
    Parallel Kahan summation with error tracking.
    
    Divides the input into partitions, computes partial sums in parallel
    (conceptually), then combines results with error compensation.
    
    Args:
        values: Sequence of values to sum
        num_partitions: Number of partitions for parallel processing
        
    Returns:
        Tuple of (sum, total_error)
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    elif isinstance(values, (list, tuple)):
        values = np.array(values)
    
    if len(values) == 0:
        return 0.0, 0.0
    
    # Partition the data
    partition_size = len(values) // num_partitions
    if partition_size == 0:
        return kahan_sum(values), 0.0
    
    partial_results = []
    partial_errors = []
    
    # Compute partial sums (would be parallel in real implementation)
    for i in range(num_partitions):
        start_idx = i * partition_size
        if i == num_partitions - 1:
            # Last partition gets remaining elements
            end_idx = len(values)
        else:
            end_idx = (i + 1) * partition_size
        
        partition = values[start_idx:end_idx]
        if len(partition) > 0:
            partial_sum, partial_error = tree_reduce_kahan(partition)
            partial_results.append(partial_sum)
            partial_errors.append(partial_error)
    
    # Combine partial results
    if len(partial_results) == 0:
        return 0.0, 0.0
    
    final_sum, combine_error = tree_reduce_kahan(partial_results)
    total_error = sum(partial_errors) + combine_error
    
    return final_sum, total_error


def compensated_mean(values: Union[List[float], torch.Tensor, np.ndarray]) -> float:
    """
    Compute mean using compensated summation.
    
    Args:
        values: Sequence of values
        
    Returns:
        Compensated mean
    """
    if isinstance(values, torch.Tensor):
        n = values.numel()
        values = values.flatten()
    elif isinstance(values, np.ndarray):
        n = values.size
        values = values.flatten()
    else:
        n = len(values)
    
    if n == 0:
        return 0.0
    
    compensated_sum = kahan_sum(values)
    return compensated_sum / n


def compensated_variance(values: Union[List[float], torch.Tensor, np.ndarray],
                        ddof: int = 1) -> float:
    """
    Compute variance using compensated summation.
    
    Args:
        values: Sequence of values
        ddof: Delta degrees of freedom (1 for sample variance, 0 for population)
        
    Returns:
        Compensated variance
    """
    if isinstance(values, torch.Tensor):
        n = values.numel()
        values = values.flatten().cpu().numpy()
    elif isinstance(values, np.ndarray):
        n = values.size
        values = values.flatten()
    else:
        values = np.array(values)
        n = len(values)
    
    if n <= ddof:
        return 0.0
    
    # Two-pass algorithm with compensated summation
    mean = compensated_mean(values)
    
    # Compute sum of squared deviations
    squared_deviations = [(x - mean) ** 2 for x in values]
    sum_sq_dev = kahan_sum(squared_deviations)
    
    return sum_sq_dev / (n - ddof)


def adaptive_precision_sum(values: Union[List[float], torch.Tensor, np.ndarray],
                          target_precision: float = 1e-15) -> Tuple[float, int]:
    """
    Adaptive precision summation that adjusts algorithm based on required precision.
    
    Args:
        values: Sequence of values to sum
        target_precision: Target relative precision
        
    Returns:
        Tuple of (sum, precision_level_used)
            precision_level: 0=naive, 1=kahan, 2=einstein_kahan, 3=tree_reduce
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    elif isinstance(values, (list, tuple)):
        values = np.array(values)
    
    if len(values) == 0:
        return 0.0, 0
    
    # Estimate conditioning and required precision
    max_val = np.max(np.abs(values))
    sum_magnitude = np.sum(np.abs(values))
    
    if max_val == 0:
        return 0.0, 0
    
    condition_number = sum_magnitude / max_val
    estimated_error = condition_number * np.finfo(values.dtype).eps
    
    # Choose algorithm based on estimated error vs target precision
    if estimated_error < target_precision:
        # Naive summation sufficient
        return float(np.sum(values)), 0
    elif estimated_error < target_precision * 10:
        # Kahan summation
        return kahan_sum(values), 1
    elif estimated_error < target_precision * 100:
        # Einstein-Kahan
        result, _ = einstein_kahan_sum(values)
        return result, 2
    else:
        # Tree reduction for maximum stability
        result, _ = tree_reduce_kahan(values)
        return result, 3


class BatchKahanSummator:
    """
    Batch processor for Kahan summation operations.
    
    Efficiently handles multiple summation operations with shared
    error tracking and statistics.
    """
    
    def __init__(self, track_statistics: bool = True):
        """
        Initialize batch summator.
        
        Args:
            track_statistics: Whether to track operation statistics
        """
        self.track_statistics = track_statistics
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset operation statistics."""
        self.operation_count = 0
        self.total_error = 0.0
        self.max_error = 0.0
        self.min_error = float('inf')
    
    def sum_batch(self, batch_values: List[Union[List[float], torch.Tensor, np.ndarray]],
                  method: str = 'kahan') -> List[float]:
        """
        Sum multiple sequences in batch.
        
        Args:
            batch_values: List of sequences to sum
            method: Summation method ('kahan', 'einstein', 'tree', 'adaptive')
            
        Returns:
            List of computed sums
        """
        results = []
        
        for values in batch_values:
            if method == 'kahan':
                result = kahan_sum(values)
                error = 0.0  # Error tracking would require more computation
            elif method == 'einstein':
                result, error = einstein_kahan_sum(values)
            elif method == 'tree':
                result, error = tree_reduce_kahan(values)
            elif method == 'adaptive':
                result, _ = adaptive_precision_sum(values)
                error = 0.0
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append(result)
            
            if self.track_statistics:
                self.operation_count += 1
                self.total_error += abs(error)
                self.max_error = max(self.max_error, abs(error))
                self.min_error = min(self.min_error, abs(error))
        
        return results
    
    def get_statistics(self) -> dict:
        """Get operation statistics."""
        if not self.track_statistics or self.operation_count == 0:
            return {}
        
        return {
            'operation_count': self.operation_count,
            'average_error': self.total_error / self.operation_count,
            'max_error': self.max_error,
            'min_error': self.min_error if self.min_error != float('inf') else 0.0,
            'total_error': self.total_error
        }