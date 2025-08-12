#!/usr/bin/env python3
"""
Basic usage examples for the Kahan Summation Library.

This script demonstrates the fundamental algorithms and their benefits
for numerical stability in floating-point arithmetic.
"""

import numpy as np
import time
from typing import List, Tuple

# Import the Kahan summation library
import sys
sys.path.append('..')

from kahan import (
    kahan_sum,
    einstein_kahan_sum, 
    tree_reduce_kahan,
    parallel_kahan_sum,
    adaptive_precision_sum,
    compensated_mean,
    compensated_variance,
    KahanAccumulator,
    BatchKahanSummator
)


def demonstrate_precision_loss():
    """Show how standard summation loses precision."""
    print("=" * 60)
    print("DEMONSTRATION: Precision Loss in Standard Summation")
    print("=" * 60)
    
    # Create test case that highlights precision issues
    # Adding small number to large number repeatedly
    large_number = 1e8
    small_number = 1.0
    
    data = np.array([large_number, small_number, -large_number], dtype=np.float32)
    
    print(f"Test data: [{large_number}, {small_number}, {-large_number}]")
    print(f"Expected result: {small_number}")
    print()
    
    # Standard NumPy summation
    naive_result = np.sum(data)
    print(f"NumPy sum result:     {naive_result}")
    print(f"Error:                {abs(naive_result - small_number):.2e}")
    print()
    
    # Kahan summation
    kahan_result = kahan_sum(data)
    print(f"Kahan sum result:     {kahan_result}")
    print(f"Error:                {abs(kahan_result - small_number):.2e}")
    print()
    
    # Einstein-Kahan summation
    einstein_result, curvature = einstein_kahan_sum(data)
    print(f"Einstein-Kahan result: {einstein_result}")
    print(f"Curvature error:       {curvature:.2e}")
    print(f"Error:                 {abs(einstein_result - small_number):.2e}")
    print()


def demonstrate_large_array_summation():
    """Demonstrate summation of large arrays with different algorithms."""
    print("=" * 60)
    print("DEMONSTRATION: Large Array Summation")
    print("=" * 60)
    
    # Create large array with mixed magnitudes
    np.random.seed(42)
    n = 1000000
    
    # Mix of large and small values to challenge precision
    large_values = np.random.normal(1e6, 1e5, n//2).astype(np.float32)
    small_values = np.random.normal(0, 1, n//2).astype(np.float32)
    data = np.concatenate([large_values, small_values])
    np.random.shuffle(data)
    
    print(f"Array size: {len(data):,}")
    print(f"Value range: [{data.min():.2e}, {data.max():.2e}]")
    print()
    
    # Compute reference in higher precision
    reference = np.sum(data.astype(np.float64))
    
    algorithms = [
        ("NumPy sum", lambda x: np.sum(x)),
        ("Kahan sum", kahan_sum),
        ("Einstein-Kahan", lambda x: einstein_kahan_sum(x)[0]),
        ("Tree reduction", lambda x: tree_reduce_kahan(x)[0]),
        ("Parallel Kahan", lambda x: parallel_kahan_sum(x)[0]),
    ]
    
    print(f"{'Algorithm':<20} {'Time (ms)':<12} {'Relative Error':<15}")
    print("-" * 50)
    
    for name, algorithm in algorithms:
        start_time = time.time()
        result = algorithm(data)
        elapsed = (time.time() - start_time) * 1000
        
        relative_error = abs(result - reference) / abs(reference)
        
        print(f"{name:<20} {elapsed:8.2f}     {relative_error:.2e}")
    
    print()


def demonstrate_incremental_summation():
    """Show incremental summation with KahanAccumulator."""
    print("=" * 60)
    print("DEMONSTRATION: Incremental Summation")
    print("=" * 60)
    
    # Simulate streaming data
    acc = KahanAccumulator()
    
    # Add values incrementally
    values = [1e8, 1.0, 2.0, 3.0, -1e8, 4.0, 5.0]
    
    print("Adding values incrementally:")
    print(f"{'Value':<15} {'Running Sum':<15} {'Error':<15}")
    print("-" * 45)
    
    for value in values:
        acc.add(value)
        print(f"{value:<15.1f} {acc.get():<15.6f} {acc.get_error():<15.2e}")
    
    print()
    print(f"Final sum: {acc.get()}")
    print(f"Expected:  {sum(values)}")
    print(f"Accumulated error: {acc.get_error():.2e}")
    print()


def demonstrate_batch_processing():
    """Show batch processing capabilities."""
    print("=" * 60)
    print("DEMONSTRATION: Batch Processing")
    print("=" * 60)
    
    # Create multiple arrays to process
    np.random.seed(123)
    arrays = [
        np.random.randn(1000).astype(np.float32),
        np.random.randn(2000).astype(np.float32),
        np.random.randn(1500).astype(np.float32),
        np.random.randn(800).astype(np.float32),
        np.random.randn(1200).astype(np.float32),
    ]
    
    print(f"Processing {len(arrays)} arrays")
    print(f"Sizes: {[len(arr) for arr in arrays]}")
    print()
    
    # Use batch processor
    batch_processor = BatchKahanSummator(track_statistics=True)
    
    # Process with different methods
    methods = ['kahan', 'einstein', 'tree']
    
    for method in methods:
        start_time = time.time()
        results = batch_processor.sum_batch(arrays, method=method)
        elapsed = (time.time() - start_time) * 1000
        
        print(f"{method.capitalize()} method:")
        print(f"  Time: {elapsed:.2f} ms")
        print(f"  Results: {[f'{r:.6f}' for r in results[:3]]}...")
        print()
    
    # Show statistics
    stats = batch_processor.get_statistics()
    if stats:
        print("Processing statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2e}")
        print()


def demonstrate_adaptive_precision():
    """Show adaptive precision algorithm selection."""
    print("=" * 60) 
    print("DEMONSTRATION: Adaptive Precision")
    print("=" * 60)
    
    # Test cases with different conditioning
    test_cases = [
        ("Well-conditioned", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("Moderate conditioning", np.array([1e3, 1.0, 2.0, -1e3])),
        ("Poor conditioning", np.array([1e8, 1.0, -1e8])),
        ("Very poor conditioning", np.array([1e15, 1.0, -1e15])),
    ]
    
    print(f"{'Test Case':<25} {'Algorithm':<15} {'Result':<15}")
    print("-" * 55)
    
    algorithm_names = {0: "Naive", 1: "Kahan", 2: "Einstein", 3: "Tree"}
    
    for name, data in test_cases:
        result, algorithm_id = adaptive_precision_sum(data.astype(np.float32))
        algorithm_name = algorithm_names[algorithm_id]
        
        print(f"{name:<25} {algorithm_name:<15} {result:<15.6f}")
    
    print()


def demonstrate_statistical_functions():
    """Show compensated statistical computations."""
    print("=" * 60)
    print("DEMONSTRATION: Compensated Statistics")
    print("=" * 60)
    
    # Create test data with challenging properties
    np.random.seed(456)
    data = np.concatenate([
        np.random.normal(1e6, 1, 1000),   # Large values
        np.random.normal(0, 1e-3, 1000),  # Small values
    ]).astype(np.float32)
    
    print(f"Data size: {len(data)}")
    print(f"Range: [{data.min():.2e}, {data.max():.2e}]")
    print()
    
    # Compare standard vs compensated statistics
    print(f"{'Statistic':<20} {'Standard':<15} {'Compensated':<15} {'Difference':<15}")
    print("-" * 65)
    
    # Mean
    std_mean = np.mean(data)
    comp_mean = compensated_mean(data)
    print(f"{'Mean':<20} {std_mean:<15.6f} {comp_mean:<15.6f} {abs(std_mean-comp_mean):<15.2e}")
    
    # Variance
    std_var = np.var(data, ddof=1)
    comp_var = compensated_variance(data, ddof=1)
    print(f"{'Variance':<20} {std_var:<15.6f} {comp_var:<15.6f} {abs(std_var-comp_var):<15.2e}")
    
    print()


def performance_comparison():
    """Compare performance across different array sizes."""
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    sizes = [1000, 10000, 100000, 1000000]
    
    print(f"{'Size':<10} {'NumPy':<10} {'Kahan':<10} {'Einstein':<10} {'Tree':<10}")
    print("-" * 50)
    
    for size in sizes:
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(size).astype(np.float32)
        
        # Benchmark each algorithm
        times = {}
        
        # NumPy
        start = time.time()
        np.sum(data)
        times['NumPy'] = (time.time() - start) * 1000
        
        # Kahan
        start = time.time()
        kahan_sum(data)
        times['Kahan'] = (time.time() - start) * 1000
        
        # Einstein-Kahan
        start = time.time()
        einstein_kahan_sum(data)
        times['Einstein'] = (time.time() - start) * 1000
        
        # Tree reduction
        start = time.time()
        tree_reduce_kahan(data)
        times['Tree'] = (time.time() - start) * 1000
        
        print(f"{size:<10} {times['NumPy']:<10.2f} {times['Kahan']:<10.2f} "
              f"{times['Einstein']:<10.2f} {times['Tree']:<10.2f}")
    
    print("\nTimes in milliseconds")
    print()


def main():
    """Run all demonstrations."""
    print("KAHAN SUMMATION LIBRARY - BASIC USAGE EXAMPLES")
    print("=" * 60)
    print()
    
    demonstrate_precision_loss()
    demonstrate_large_array_summation()
    demonstrate_incremental_summation()
    demonstrate_batch_processing() 
    demonstrate_adaptive_precision()
    demonstrate_statistical_functions()
    performance_comparison()
    
    print("=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()