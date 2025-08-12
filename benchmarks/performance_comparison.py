#!/usr/bin/env python3
"""
Performance comparison benchmarks for Kahan summation algorithms.

This script measures execution time, memory usage, and scalability
of different summation algorithms across various scenarios.
"""

import numpy as np
import time
import psutil
import gc
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
import multiprocessing as mp
import sys
sys.path.append('..')

from kahan import (
    kahan_sum,
    einstein_kahan_sum,
    tree_reduce_kahan,
    parallel_kahan_sum,
    adaptive_precision_sum,
    BatchKahanSummator
)


class PerformanceBenchmark:
    """
    Comprehensive performance benchmark suite for summation algorithms.
    """
    
    def __init__(self):
        self.algorithms = {
            'numpy': np.sum,
            'kahan': kahan_sum,
            'einstein': lambda x: einstein_kahan_sum(x)[0],
            'tree': lambda x: tree_reduce_kahan(x)[0],
            'parallel': lambda x: parallel_kahan_sum(x)[0],
            'adaptive': lambda x: adaptive_precision_sum(x)[0],
        }
        
        self.results = []
    
    def measure_memory_usage(self, func: Callable, data: np.ndarray) -> Tuple[float, float]:
        """
        Measure peak memory usage during function execution.
        
        Args:
            func: Function to benchmark
            data: Input data
            
        Returns:
            Tuple of (peak_memory_mb, memory_increase_mb)
        """
        process = psutil.Process()
        
        # Force garbage collection before measurement
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(data)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        return peak_memory, memory_increase
    
    def benchmark_execution_time(self, func: Callable, data: np.ndarray, 
                                num_runs: int = 5) -> Dict:
        """
        Benchmark execution time with multiple runs for statistical accuracy.
        
        Args:
            func: Function to benchmark
            data: Input data
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with timing statistics
        """
        times = []
        
        for _ in range(num_runs):
            # Warm up
            _ = func(data[:min(1000, len(data))])
            
            # Actual measurement
            start_time = time.perf_counter()
            result = func(data)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
        }
    
    def run_scalability_benchmark(self) -> pd.DataFrame:
        """
        Test performance scalability with increasing array sizes.
        
        Returns:
            DataFrame with scalability results
        """
        print("Running scalability benchmark...")
        
        # Test array sizes
        sizes = [10**i for i in range(2, 8)]  # 100 to 10M elements
        dtypes = [np.float32, np.float64]
        
        results = []
        
        for dtype in dtypes:
            print(f"\nTesting {dtype.__name__} arrays:")
            
            for size in sizes:
                print(f"  Size: {size:,}")
                
                # Generate test data
                np.random.seed(42)
                data = np.random.normal(0, 1, size).astype(dtype)
                
                for alg_name, algorithm in self.algorithms.items():
                    try:
                        # Benchmark timing
                        timing_stats = self.benchmark_execution_time(algorithm, data)
                        
                        # Measure memory
                        peak_mem, mem_increase = self.measure_memory_usage(algorithm, data)
                        
                        # Calculate throughput
                        throughput = size / timing_stats['mean_time'] / 1e6  # Million elements per second
                        
                        result = {
                            'algorithm': alg_name,
                            'size': size,
                            'dtype': dtype.__name__,
                            'throughput_meps': throughput,
                            'peak_memory_mb': peak_mem,
                            'memory_increase_mb': mem_increase,
                            **timing_stats
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"    Error in {alg_name}: {e}")
                        continue
        
        return pd.DataFrame(results)
    
    def run_data_type_benchmark(self) -> pd.DataFrame:
        """
        Compare performance across different data types and patterns.
        
        Returns:
            DataFrame with data type performance results
        """
        print("\nRunning data type benchmark...")
        
        size = 100000
        data_patterns = {
            'random_normal': lambda: np.random.normal(0, 1, size),
            'random_uniform': lambda: np.random.uniform(-1, 1, size),
            'alternating': lambda: np.array([1.0, -1.0] * (size // 2)),
            'geometric_series': lambda: 0.5 ** np.arange(size),
            'large_values': lambda: np.random.normal(1e6, 1e5, size),
            'small_values': lambda: np.random.normal(0, 1e-6, size),
            'mixed_magnitude': lambda: np.concatenate([
                np.random.normal(1e6, 1e5, size//2),
                np.random.normal(0, 1e-6, size//2)
            ]),
        }
        
        dtypes = [np.float16, np.float32, np.float64]
        results = []
        
        for pattern_name, pattern_func in data_patterns.items():
            print(f"  Pattern: {pattern_name}")
            
            for dtype in dtypes:
                # Generate data
                np.random.seed(42)
                data = pattern_func().astype(dtype)
                
                for alg_name, algorithm in self.algorithms.items():
                    try:
                        timing_stats = self.benchmark_execution_time(algorithm, data)
                        peak_mem, mem_increase = self.measure_memory_usage(algorithm, data)
                        
                        result = {
                            'algorithm': alg_name,
                            'pattern': pattern_name,
                            'dtype': dtype.__name__,
                            'size': size,
                            'peak_memory_mb': peak_mem,
                            'memory_increase_mb': mem_increase,
                            **timing_stats
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"    Error in {alg_name} with {dtype.__name__}: {e}")
                        continue
        
        return pd.DataFrame(results)
    
    def run_parallel_scalability_benchmark(self) -> pd.DataFrame:
        """
        Test parallel algorithm scalability with different thread counts.
        
        Returns:
            DataFrame with parallel scalability results
        """
        print("\nRunning parallel scalability benchmark...")
        
        size = 1000000
        max_threads = min(mp.cpu_count(), 16)
        thread_counts = [1, 2, 4, 8, max_threads]
        
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(0, 1, size).astype(np.float32)
        
        results = []
        
        # Baseline with single-threaded algorithms
        baseline_algorithms = {
            'numpy': np.sum,
            'kahan': kahan_sum,
            'tree_st': lambda x: tree_reduce_kahan(x)[0],  # Single-threaded tree
        }
        
        for alg_name, algorithm in baseline_algorithms.items():
            timing_stats = self.benchmark_execution_time(algorithm, data)
            
            result = {
                'algorithm': alg_name,
                'threads': 1,
                'size': size,
                'speedup': 1.0,
                'efficiency': 1.0,
                **timing_stats
            }
            results.append(result)
        
        # Test parallel algorithms with different thread counts
        baseline_time = results[0]['mean_time']  # Use numpy as baseline
        
        for num_threads in thread_counts:
            if num_threads == 1:
                continue
                
            print(f"  Testing with {num_threads} threads")
            
            # Parallel Kahan sum
            parallel_func = lambda x: parallel_kahan_sum(x, num_partitions=num_threads)[0]
            timing_stats = self.benchmark_execution_time(parallel_func, data)
            
            speedup = baseline_time / timing_stats['mean_time']
            efficiency = speedup / num_threads
            
            result = {
                'algorithm': 'parallel_kahan',
                'threads': num_threads,
                'size': size,
                'speedup': speedup,
                'efficiency': efficiency,
                **timing_stats
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def run_batch_processing_benchmark(self) -> pd.DataFrame:
        """
        Benchmark batch processing capabilities.
        
        Returns:
            DataFrame with batch processing results
        """
        print("\nRunning batch processing benchmark...")
        
        # Test parameters
        array_sizes = [100, 1000, 10000]
        batch_sizes = [10, 100, 1000]
        
        results = []
        
        for array_size in array_sizes:
            for batch_size in batch_sizes:
                print(f"  Array size: {array_size}, Batch size: {batch_size}")
                
                # Generate batch data
                np.random.seed(42)
                batch_data = [
                    np.random.normal(0, 1, array_size).astype(np.float32)
                    for _ in range(batch_size)
                ]
                
                # Test individual processing
                start_time = time.perf_counter()
                individual_results = [kahan_sum(arr) for arr in batch_data]
                individual_time = time.perf_counter() - start_time
                
                # Test batch processing
                batch_processor = BatchKahanSummator()
                
                for method in ['kahan', 'einstein', 'tree']:
                    start_time = time.perf_counter()
                    batch_results = batch_processor.sum_batch(batch_data, method=method)
                    batch_time = time.perf_counter() - start_time
                    
                    speedup = individual_time / batch_time if batch_time > 0 else 0
                    
                    result = {
                        'array_size': array_size,
                        'batch_size': batch_size,
                        'method': method,
                        'individual_time': individual_time,
                        'batch_time': batch_time,
                        'speedup': speedup,
                        'total_elements': array_size * batch_size,
                    }
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_scalability_results(self, df: pd.DataFrame) -> None:
        """
        Analyze and display scalability benchmark results.
        
        Args:
            df: DataFrame with scalability results
        """
        print("\n" + "="*80)
        print("SCALABILITY ANALYSIS")
        print("="*80)
        
        # Throughput comparison
        print("\nTHROUGHPUT COMPARISON (Million Elements Per Second):")
        print("-" * 60)
        
        # Get largest size results for fair comparison
        max_size = df['size'].max()
        large_df = df[df['size'] == max_size]
        
        print(f"Array size: {max_size:,} elements")
        print(f"{'Algorithm':<15} {'float32 MEPS':<15} {'float64 MEPS':<15}")
        print("-" * 50)
        
        for alg in df['algorithm'].unique():
            alg_df = large_df[large_df['algorithm'] == alg]
            
            f32_throughput = alg_df[alg_df['dtype'] == 'float32']['throughput_meps'].iloc[0] \
                           if len(alg_df[alg_df['dtype'] == 'float32']) > 0 else 0
            f64_throughput = alg_df[alg_df['dtype'] == 'float64']['throughput_meps'].iloc[0] \
                           if len(alg_df[alg_df['dtype'] == 'float64']) > 0 else 0
            
            print(f"{alg:<15} {f32_throughput:<15.2f} {f64_throughput:<15.2f}")
        
        # Memory usage analysis
        print("\nMEMORY USAGE ANALYSIS:")
        print("-" * 30)
        print(f"{'Algorithm':<15} {'Peak Memory (MB)':<18} {'Memory Increase (MB)':<20}")
        print("-" * 55)
        
        for alg in df['algorithm'].unique():
            alg_df = large_df[large_df['algorithm'] == alg]
            avg_peak = alg_df['peak_memory_mb'].mean()
            avg_increase = alg_df['memory_increase_mb'].mean()
            
            print(f"{alg:<15} {avg_peak:<18.1f} {avg_increase:<20.1f}")
        
        # Scaling behavior
        print("\nSCALING BEHAVIOR (Time vs Size):")
        print("-" * 35)
        
        for dtype in df['dtype'].unique():
            print(f"\n{dtype} arrays:")
            
            for alg in df['algorithm'].unique():
                alg_df = df[(df['algorithm'] == alg) & (df['dtype'] == dtype)]
                
                if len(alg_df) >= 2:
                    # Calculate scaling exponent
                    sizes = alg_df['size'].values
                    times = alg_df['mean_time'].values
                    
                    # Linear regression in log space: log(time) = a * log(size) + b
                    log_sizes = np.log10(sizes)
                    log_times = np.log10(times)
                    
                    if len(log_sizes) > 1:
                        slope = np.polyfit(log_sizes, log_times, 1)[0]
                        print(f"  {alg}: O(n^{slope:.2f})")
    
    def plot_performance_results(self, scalability_df: pd.DataFrame, 
                               parallel_df: pd.DataFrame = None,
                               save_plots: bool = True) -> None:
        """
        Create performance visualization plots.
        
        Args:
            scalability_df: Scalability benchmark results
            parallel_df: Parallel scalability results (optional)
            save_plots: Whether to save plots to files
        """
        # Plot 1: Throughput vs Array Size
        plt.figure(figsize=(12, 8))
        
        algorithms = scalability_df['algorithm'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        
        for i, alg in enumerate(algorithms):
            alg_df = scalability_df[
                (scalability_df['algorithm'] == alg) & 
                (scalability_df['dtype'] == 'float32')
            ]
            
            if len(alg_df) > 0:
                plt.loglog(alg_df['size'], alg_df['throughput_meps'], 
                          'o-', color=colors[i], label=alg, markersize=6)
        
        plt.xlabel('Array Size')
        plt.ylabel('Throughput (Million Elements/sec)')
        plt.title('Performance Scalability - float32')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('throughput_scalability.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Execution Time vs Array Size
        plt.figure(figsize=(12, 8))
        
        for i, alg in enumerate(algorithms):
            alg_df = scalability_df[
                (scalability_df['algorithm'] == alg) & 
                (scalability_df['dtype'] == 'float32')
            ]
            
            if len(alg_df) > 0:
                plt.loglog(alg_df['size'], alg_df['mean_time'] * 1000, 
                          'o-', color=colors[i], label=alg, markersize=6)
        
        plt.xlabel('Array Size')
        plt.ylabel('Execution Time (ms)')
        plt.title('Execution Time Scalability - float32')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('time_scalability.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Memory Usage
        plt.figure(figsize=(10, 6))
        
        # Use largest size for memory comparison
        max_size = scalability_df['size'].max()
        memory_df = scalability_df[
            (scalability_df['size'] == max_size) & 
            (scalability_df['dtype'] == 'float32')
        ]
        
        algorithms = memory_df['algorithm'].tolist()
        memory_usage = memory_df['memory_increase_mb'].tolist()
        
        plt.bar(algorithms, memory_usage, color=colors[:len(algorithms)], alpha=0.8)
        plt.xlabel('Algorithm')
        plt.ylabel('Memory Increase (MB)')
        plt.title(f'Memory Usage Comparison - {max_size:,} elements')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('memory_usage.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 4: Parallel Scalability (if available)
        if parallel_df is not None and len(parallel_df) > 0:
            plt.figure(figsize=(12, 5))
            
            # Speedup plot
            plt.subplot(1, 2, 1)
            parallel_kahan = parallel_df[parallel_df['algorithm'] == 'parallel_kahan']
            
            if len(parallel_kahan) > 0:
                plt.plot(parallel_kahan['threads'], parallel_kahan['speedup'], 
                        'o-', label='Parallel Kahan', markersize=8)
                plt.plot([1, parallel_kahan['threads'].max()], 
                        [1, parallel_kahan['threads'].max()], 
                        '--', color='gray', label='Ideal Speedup')
                
                plt.xlabel('Number of Threads')
                plt.ylabel('Speedup')
                plt.title('Parallel Speedup')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Efficiency plot
            plt.subplot(1, 2, 2)
            
            if len(parallel_kahan) > 0:
                plt.plot(parallel_kahan['threads'], parallel_kahan['efficiency'] * 100, 
                        'o-', color='red', label='Parallel Efficiency', markersize=8)
                plt.axhline(y=100, color='gray', linestyle='--', label='100% Efficiency')
                
                plt.xlabel('Number of Threads')
                plt.ylabel('Efficiency (%)')
                plt.title('Parallel Efficiency')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('parallel_scalability.png', dpi=300, bbox_inches='tight')
            plt.show()


def main():
    """Run the complete performance benchmark suite."""
    print("KAHAN SUMMATION LIBRARY - PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    print("Starting performance benchmarks...")
    
    scalability_df = benchmark.run_scalability_benchmark()
    datatype_df = benchmark.run_data_type_benchmark()
    parallel_df = benchmark.run_parallel_scalability_benchmark()
    batch_df = benchmark.run_batch_processing_benchmark()
    
    # Save results
    scalability_df.to_csv('scalability_benchmark.csv', index=False)
    datatype_df.to_csv('datatype_benchmark.csv', index=False)
    parallel_df.to_csv('parallel_benchmark.csv', index=False)
    batch_df.to_csv('batch_benchmark.csv', index=False)
    
    print("\nResults saved to CSV files")
    
    # Analyze results
    benchmark.analyze_scalability_results(scalability_df)
    
    # Create plots
    try:
        benchmark.plot_performance_results(scalability_df, parallel_df)
    except ImportError:
        print("\nMatplotlib not available, skipping plots")
    except Exception as e:
        print(f"\nError creating plots: {e}")
    
    print("\n" + "="*60)
    print("Performance benchmark completed!")
    print("="*60)


if __name__ == "__main__":
    main()