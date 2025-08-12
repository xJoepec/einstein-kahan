#!/usr/bin/env python3
"""
Unit tests for Kahan summation algorithms.

Tests the high-level algorithms in the kahan.algorithms module.
"""

import pytest
import numpy as np
import torch
from typing import List
import sys
sys.path.append('..')

from kahan.algorithms import (
    kahan_sum,
    einstein_kahan_sum,
    tree_reduce_kahan,
    parallel_kahan_sum,
    compensated_mean,
    compensated_variance,
    adaptive_precision_sum,
    BatchKahanSummator
)


class TestKahanSum:
    """Test cases for basic kahan_sum function."""
    
    def test_basic_functionality(self):
        """Test basic Kahan summation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = kahan_sum(values)
        expected = sum(values)
        
        assert abs(result - expected) < 1e-15
    
    def test_precision_improvement(self):
        """Test precision improvement over naive summation."""
        # Challenging case: large number + small number - large number
        values = np.array([1e16, 1.0, -1e16], dtype=np.float32)
        
        naive_result = float(np.sum(values))
        kahan_result = kahan_sum(values)
        expected = 1.0
        
        # Kahan should be more accurate
        naive_error = abs(naive_result - expected)
        kahan_error = abs(kahan_result - expected)
        
        assert kahan_error < naive_error
    
    def test_different_input_types(self):
        """Test with different input types."""
        # List
        result1 = kahan_sum([1.0, 2.0, 3.0])
        
        # NumPy array
        result2 = kahan_sum(np.array([1.0, 2.0, 3.0]))
        
        # PyTorch tensor
        result3 = kahan_sum(torch.tensor([1.0, 2.0, 3.0]))
        
        # All should be equal
        assert abs(result1 - 6.0) < 1e-15
        assert abs(result2 - 6.0) < 1e-15
        assert abs(result3 - 6.0) < 1e-15
    
    def test_empty_input(self):
        """Test with empty input."""
        result = kahan_sum([])
        assert result == 0.0
    
    def test_single_element(self):
        """Test with single element."""
        value = 42.5
        result = kahan_sum([value])
        assert result == value
    
    def test_large_array(self):
        """Test with large array."""
        n = 100000
        np.random.seed(42)
        values = np.random.randn(n).astype(np.float32)
        
        kahan_result = kahan_sum(values)
        
        # Compare with higher precision computation
        expected = np.sum(values.astype(np.float64))
        relative_error = abs(kahan_result - expected) / abs(expected)
        
        assert relative_error < 1e-6


class TestEinsteinKahanSum:
    """Test cases for Einstein-Kahan summation."""
    
    def test_basic_functionality(self):
        """Test basic Einstein-Kahan summation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result, curvature = einstein_kahan_sum(values)
        expected = sum(values)
        
        assert abs(result - expected) < 1e-15
        assert isinstance(curvature, float)
    
    def test_geometric_correction(self):
        """Test geometric correction with challenging case."""
        # Case with significant floating-point error
        values = np.array([1e8, 1.0, -1e8], dtype=np.float32)
        
        result, curvature = einstein_kahan_sum(values)
        expected = 1.0
        
        assert abs(result - expected) < 1e-6
        assert abs(curvature) > 0  # Should detect curvature error
    
    def test_custom_einstein_constant(self):
        """Test with custom Einstein constant."""
        values = [1.0, 2.0, 3.0]
        custom_constant = 4.0 * np.pi
        
        result1, _ = einstein_kahan_sum(values)
        result2, _ = einstein_kahan_sum(values, einstein_constant=custom_constant)
        
        # Results may differ due to different correction factors
        # But both should be reasonable
        expected = sum(values)
        assert abs(result1 - expected) < 1e-10
        assert abs(result2 - expected) < 1e-10
    
    def test_path_independence(self):
        """Test that Einstein-Kahan gives path-independent results."""
        values = np.array([1e6, 1.0, 2.0, -1e6, 3.0], dtype=np.float32)
        
        # Try different orderings
        result1, _ = einstein_kahan_sum(values)
        result2, _ = einstein_kahan_sum(values[::-1])  # Reversed
        result3, _ = einstein_kahan_sum(np.random.permutation(values))
        
        # Should be very close (path-independent)
        assert abs(result1 - result2) < 1e-6
        assert abs(result1 - result3) < 1e-6


class TestTreeReduceKahan:
    """Test cases for tree reduction algorithm."""
    
    def test_basic_functionality(self):
        """Test basic tree reduction."""
        values = list(range(1, 9))  # [1, 2, 3, 4, 5, 6, 7, 8]
        result, error = tree_reduce_kahan(values)
        expected = sum(values)
        
        assert abs(result - expected) < 1e-15
        assert isinstance(error, float)
    
    def test_power_of_two_sizes(self):
        """Test with power-of-two sizes (optimal for tree reduction)."""
        for n in [2, 4, 8, 16, 32, 64]:
            values = list(range(n))
            result, error = tree_reduce_kahan(values)
            expected = sum(values)
            
            assert abs(result - expected) < 1e-12
    
    def test_non_power_of_two_sizes(self):
        """Test with non-power-of-two sizes."""
        for n in [3, 5, 7, 10, 15, 31]:
            values = list(range(n))
            result, error = tree_reduce_kahan(values)
            expected = sum(values)
            
            assert abs(result - expected) < 1e-12
    
    def test_precision_improvement(self):
        """Test precision improvement over standard summation."""
        # Create challenging case for tree reduction
        n = 1024
        np.random.seed(42)
        
        # Mix of large and small values
        large_vals = np.random.normal(1e6, 1e5, n//2)
        small_vals = np.random.normal(0, 1, n//2)
        values = np.concatenate([large_vals, small_vals]).astype(np.float32)
        np.random.shuffle(values)
        
        tree_result, tree_error = tree_reduce_kahan(values)
        naive_result = float(np.sum(values))
        
        # Compare with high-precision reference
        reference = np.sum(values.astype(np.float64))
        
        tree_rel_error = abs(tree_result - reference) / abs(reference)
        naive_rel_error = abs(naive_result - reference) / abs(reference)
        
        assert tree_rel_error <= naive_rel_error
    
    def test_empty_and_single_element(self):
        """Test edge cases."""
        # Empty
        result, error = tree_reduce_kahan([])
        assert result == 0.0
        assert error == 0.0
        
        # Single element
        result, error = tree_reduce_kahan([42.5])
        assert result == 42.5
        assert error == 0.0


class TestParallelKahanSum:
    """Test cases for parallel Kahan summation."""
    
    def test_basic_functionality(self):
        """Test basic parallel summation."""
        values = list(range(100))
        result, error = parallel_kahan_sum(values, num_partitions=4)
        expected = sum(values)
        
        assert abs(result - expected) < 1e-12
        assert isinstance(error, float)
    
    def test_different_partition_counts(self):
        """Test with different numbers of partitions."""
        values = list(range(1000))
        expected = sum(values)
        
        for num_partitions in [1, 2, 4, 8, 16]:
            result, error = parallel_kahan_sum(values, num_partitions=num_partitions)
            assert abs(result - expected) < 1e-10
    
    def test_partition_size_edge_cases(self):
        """Test edge cases with partition sizes."""
        # More partitions than elements
        values = [1.0, 2.0, 3.0]
        result, error = parallel_kahan_sum(values, num_partitions=10)
        assert abs(result - 6.0) < 1e-15
        
        # Single partition
        result, error = parallel_kahan_sum(values, num_partitions=1)
        assert abs(result - 6.0) < 1e-15
    
    def test_precision_with_challenging_data(self):
        """Test precision with challenging data patterns."""
        # Alternating large positive/negative values
        n = 1000
        values = []
        for i in range(n):
            if i % 2 == 0:
                values.append(1e8)
            else:
                values.append(-1e8 + 1.0)  # Net contribution of 1.0 per pair
        
        result, error = parallel_kahan_sum(values, num_partitions=8)
        expected = n // 2  # 500 pairs, each contributing 1.0
        
        assert abs(result - expected) < 1.0  # Should be reasonably close


class TestCompensatedStatistics:
    """Test cases for compensated statistical functions."""
    
    def test_compensated_mean(self):
        """Test compensated mean calculation."""
        # Simple case
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compensated_mean(values)
        expected = 3.0
        
        assert abs(result - expected) < 1e-15
        
        # Test with different input types
        result_torch = compensated_mean(torch.tensor(values))
        result_numpy = compensated_mean(np.array(values))
        
        assert abs(result_torch - expected) < 1e-15
        assert abs(result_numpy - expected) < 1e-15
    
    def test_compensated_mean_precision(self):
        """Test precision improvement in mean calculation."""
        # Large numbers that stress precision
        n = 1000
        base_value = 1e6
        small_variations = np.random.normal(0, 1, n)
        values = (base_value + small_variations).astype(np.float32)
        
        compensated_result = compensated_mean(values)
        naive_result = float(np.mean(values))
        
        # Compare with high-precision reference
        reference = np.mean(values.astype(np.float64))
        
        comp_error = abs(compensated_result - reference)
        naive_error = abs(naive_result - reference)
        
        assert comp_error <= naive_error
    
    def test_compensated_variance(self):
        """Test compensated variance calculation."""
        # Known variance case
        values = [1, 2, 3, 4, 5]
        result = compensated_variance(values, ddof=1)
        
        # Expected sample variance
        expected = np.var(values, ddof=1)
        assert abs(result - expected) < 1e-15
        
        # Test population variance
        result_pop = compensated_variance(values, ddof=0)
        expected_pop = np.var(values, ddof=0)
        assert abs(result_pop - expected_pop) < 1e-15
    
    def test_variance_edge_cases(self):
        """Test variance calculation edge cases."""
        # Single element (ddof=1)
        result = compensated_variance([5.0], ddof=1)
        assert result == 0.0
        
        # Two elements
        result = compensated_variance([1.0, 3.0], ddof=1)
        expected = 2.0  # (1-2)^2 + (3-2)^2 = 2, divided by (2-1) = 2
        assert abs(result - expected) < 1e-15
        
        # Empty array should work
        result = compensated_variance([], ddof=1)
        assert result == 0.0


class TestAdaptivePrecisionSum:
    """Test cases for adaptive precision summation."""
    
    def test_algorithm_selection(self):
        """Test that appropriate algorithms are selected."""
        # Well-conditioned case (should use naive or Kahan)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result, algorithm_id = adaptive_precision_sum(values)
        
        assert result == 15.0
        assert algorithm_id in [0, 1]  # Naive or Kahan
        
        # Poorly conditioned case (should use Einstein or Tree)
        challenging_values = np.array([1e8, 1.0, -1e8], dtype=np.float32)
        result, algorithm_id = adaptive_precision_sum(challenging_values)
        
        assert abs(result - 1.0) < 1e-6
        assert algorithm_id in [1, 2, 3]  # Kahan, Einstein, or Tree
    
    def test_different_target_precisions(self):
        """Test with different target precision requirements."""
        values = np.random.randn(1000).astype(np.float32)
        
        # High precision requirement
        result1, alg1 = adaptive_precision_sum(values, target_precision=1e-15)
        
        # Low precision requirement
        result2, alg2 = adaptive_precision_sum(values, target_precision=1e-3)
        
        # Should select more sophisticated algorithm for higher precision
        assert alg1 >= alg2
    
    def test_consistency(self):
        """Test that adaptive selection gives consistent results."""
        values = np.random.randn(10000).astype(np.float32)
        
        # Run multiple times
        results = []
        algorithms = []
        
        for _ in range(5):
            result, alg = adaptive_precision_sum(values)
            results.append(result)
            algorithms.append(alg)
        
        # Should select same algorithm and give same result
        assert len(set(algorithms)) == 1  # All same algorithm
        assert all(abs(r - results[0]) < 1e-10 for r in results)


class TestBatchKahanSummator:
    """Test cases for batch processing."""
    
    def test_basic_batch_processing(self):
        """Test basic batch summation."""
        arrays = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0]
        ]
        
        summator = BatchKahanSummator()
        results = summator.sum_batch(arrays, method='kahan')
        
        expected = [6.0, 9.0, 30.0]
        
        assert len(results) == len(expected)
        for result, exp in zip(results, expected):
            assert abs(result - exp) < 1e-15
    
    def test_different_methods(self):
        """Test different batch processing methods."""
        arrays = [
            np.random.randn(100),
            np.random.randn(200),
            np.random.randn(150)
        ]
        
        summator = BatchKahanSummator()
        
        methods = ['kahan', 'einstein', 'tree', 'adaptive']
        results = {}
        
        for method in methods:
            results[method] = summator.sum_batch(arrays, method=method)
        
        # All methods should give similar results
        for i in range(len(arrays)):
            values = [results[method][i] for method in methods]
            max_diff = max(values) - min(values)
            assert max_diff < 1e-6
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        arrays = [np.random.randn(100) for _ in range(10)]
        
        summator = BatchKahanSummator(track_statistics=True)
        _ = summator.sum_batch(arrays, method='einstein')
        
        stats = summator.get_statistics()
        
        assert 'operation_count' in stats
        assert stats['operation_count'] == 10
        assert 'average_error' in stats
        assert 'max_error' in stats
        assert 'min_error' in stats
    
    def test_statistics_reset(self):
        """Test statistics reset."""
        arrays = [np.random.randn(50) for _ in range(5)]
        
        summator = BatchKahanSummator(track_statistics=True)
        _ = summator.sum_batch(arrays)
        
        # Should have statistics
        stats1 = summator.get_statistics()
        assert stats1['operation_count'] == 5
        
        # Reset and check
        summator.reset_statistics()
        stats2 = summator.get_statistics()
        assert stats2['operation_count'] == 0
    
    def test_mixed_input_types(self):
        """Test batch processing with mixed input types."""
        arrays = [
            [1.0, 2.0, 3.0],                    # List
            np.array([4.0, 5.0]),               # NumPy array
            torch.tensor([6.0, 7.0, 8.0])       # PyTorch tensor
        ]
        
        summator = BatchKahanSummator()
        results = summator.sum_batch(arrays)
        
        expected = [6.0, 9.0, 21.0]
        
        assert len(results) == len(expected)
        for result, exp in zip(results, expected):
            assert abs(result - exp) < 1e-15


@pytest.mark.parametrize("algorithm", [
    kahan_sum,
    lambda x: einstein_kahan_sum(x)[0],
    lambda x: tree_reduce_kahan(x)[0],
    lambda x: parallel_kahan_sum(x)[0]
])
def test_algorithm_consistency(algorithm):
    """Test that all algorithms give consistent results."""
    # Test data
    np.random.seed(42)
    values = np.random.randn(1000)
    
    result = algorithm(values)
    expected = np.sum(values.astype(np.float64))
    
    relative_error = abs(result - expected) / abs(expected)
    assert relative_error < 1e-6


@pytest.mark.parametrize("size", [0, 1, 10, 100, 1000])
def test_size_robustness(size):
    """Test algorithms with different input sizes."""
    if size == 0:
        values = []
        expected = 0.0
    else:
        values = list(range(size))
        expected = size * (size - 1) // 2
    
    # Test multiple algorithms
    assert abs(kahan_sum(values) - expected) < 1e-12
    
    if size > 0:  # These functions might not handle empty inputs
        result, _ = einstein_kahan_sum(values)
        assert abs(result - expected) < 1e-12
        
        result, _ = tree_reduce_kahan(values)
        assert abs(result - expected) < 1e-12


def test_numerical_stability_comparison():
    """Compare numerical stability across algorithms."""
    # Create pathological case
    n = 1000
    large_val = 1e8
    small_vals = np.random.uniform(0, 1, n)
    
    # Pattern: [large, small1, small2, ..., -large, result_should_be_sum(small)]
    values = np.concatenate([[large_val], small_vals, [-large_val]])
    values = values.astype(np.float32)
    
    expected = np.sum(small_vals)  # True result
    
    # Test different algorithms
    naive_result = float(np.sum(values))
    kahan_result = kahan_sum(values)
    einstein_result, _ = einstein_kahan_sum(values)
    tree_result, _ = tree_reduce_kahan(values)
    
    # Calculate relative errors
    naive_error = abs(naive_result - expected) / expected
    kahan_error = abs(kahan_result - expected) / expected
    einstein_error = abs(einstein_result - expected) / expected
    tree_error = abs(tree_result - expected) / expected
    
    # Advanced algorithms should be more accurate
    assert kahan_error <= naive_error
    assert einstein_error <= naive_error
    assert tree_error <= naive_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])