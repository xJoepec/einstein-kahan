#!/usr/bin/env python3
"""
Accuracy comparison benchmarks for Kahan summation algorithms.

This script systematically tests the numerical accuracy of different
summation algorithms across various challenging test cases.
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
import sys
sys.path.append('..')

from kahan import (
    kahan_sum,
    einstein_kahan_sum,
    tree_reduce_kahan,
    parallel_kahan_sum,
    adaptive_precision_sum
)


class AccuracyBenchmark:
    """
    Comprehensive accuracy benchmark suite for summation algorithms.
    """
    
    def __init__(self):
        self.algorithms = {
            'numpy': np.sum,
            'kahan': kahan_sum,
            'einstein': lambda x: einstein_kahan_sum(x)[0],
            'tree': lambda x: tree_reduce_kahan(x)[0],
            'parallel': lambda x: parallel_kahan_sum(x)[0],
        }
        
        self.results = []
    
    def generate_test_case(self, case_type: str, size: int, dtype=np.float32) -> Tuple[np.ndarray, float]:
        """
        Generate test cases with known exact results.
        
        Args:
            case_type: Type of test case
            size: Array size
            dtype: Data type
            
        Returns:
            Tuple of (test_array, exact_result)
        """
        np.random.seed(42)  # Reproducible results
        
        if case_type == 'alternating_large':
            # Alternating large positive and negative numbers
            data = np.zeros(size, dtype=dtype)
            data[::2] = 1e8
            data[1::2] = -1e8
            if size % 2 == 1:
                data[-1] = 1.0  # Add small remainder
                exact = 1.0
            else:
                exact = 0.0
            
        elif case_type == 'geometric_series':
            # Geometric series: 1 + 1/2 + 1/4 + ... 
            powers = np.arange(size, dtype=np.float64)
            data = (0.5 ** powers).astype(dtype)
            exact = 2.0 * (1 - 0.5**size)  # Sum of geometric series
            
        elif case_type == 'harmonic_series':
            # Harmonic series: 1 + 1/2 + 1/3 + ...
            denominators = np.arange(1, size + 1, dtype=np.float64)
            data = (1.0 / denominators).astype(dtype)
            exact = np.sum(1.0 / denominators)  # Compute in high precision
            
        elif case_type == 'mixed_magnitude':
            # Mix of very large and very small numbers
            large_vals = np.full(size//2, 1e6, dtype=dtype)
            small_vals = np.random.uniform(0, 1, size//2).astype(dtype)
            data = np.concatenate([large_vals, small_vals])
            np.random.shuffle(data)
            exact = float(size//2 * 1e6 + np.sum(small_vals.astype(np.float64)))
            
        elif case_type == 'pathological_cancellation':
            # Designed to cause maximum cancellation error
            # Pattern: [1, -1+ε, 1, -1+ε, ...]
            epsilon = np.finfo(dtype).eps * 10
            data = np.zeros(size, dtype=dtype)
            data[::2] = 1.0
            data[1::2] = -1.0 + epsilon
            exact = float((size // 2) * epsilon)
            
        elif case_type == 'random_normal':
            # Random normal distribution
            data = np.random.normal(0, 1, size).astype(dtype)
            exact = np.sum(data.astype(np.float64))
            
        elif case_type == 'random_exponential':
            # Random exponential distribution (different scales)
            data = np.random.exponential(1.0, size).astype(dtype)
            exact = np.sum(data.astype(np.float64))
            
        elif case_type == 'ill_conditioned':
            # Ill-conditioned summation with large condition number
            # Create numbers spanning many orders of magnitude
            exponents = np.random.uniform(-10, 10, size)
            signs = np.random.choice([-1, 1], size)
            data = (signs * 10.0 ** exponents).astype(dtype)
            exact = np.sum(data.astype(np.float64))
            
        else:
            raise ValueError(f"Unknown test case type: {case_type}")
        
        return data, exact
    
    def run_single_benchmark(self, test_name: str, data: np.ndarray, exact: float) -> Dict:
        """
        Run benchmark on a single test case.
        
        Args:
            test_name: Name of the test case
            data: Test data
            exact: Exact result
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'test_name': test_name,
            'size': len(data),
            'exact_result': exact,
            'condition_number': self._estimate_condition_number(data),
        }
        
        for alg_name, algorithm in self.algorithms.items():
            try:
                # Time the algorithm
                start_time = time.perf_counter()
                result = algorithm(data)
                elapsed_time = time.perf_counter() - start_time
                
                # Calculate error metrics
                absolute_error = abs(result - exact)
                if exact != 0:
                    relative_error = absolute_error / abs(exact)
                else:
                    relative_error = absolute_error
                
                results[f'{alg_name}_result'] = result
                results[f'{alg_name}_time'] = elapsed_time
                results[f'{alg_name}_abs_error'] = absolute_error
                results[f'{alg_name}_rel_error'] = relative_error
                
            except Exception as e:
                print(f"Error in {alg_name} for {test_name}: {e}")
                results[f'{alg_name}_result'] = np.nan
                results[f'{alg_name}_time'] = np.nan
                results[f'{alg_name}_abs_error'] = np.inf
                results[f'{alg_name}_rel_error'] = np.inf
        
        return results
    
    def _estimate_condition_number(self, data: np.ndarray) -> float:
        """Estimate condition number for summation problem."""
        if len(data) == 0:
            return 1.0
        
        abs_sum = np.sum(np.abs(data.astype(np.float64)))
        result_sum = abs(np.sum(data.astype(np.float64)))
        
        if result_sum == 0:
            return np.inf
        return abs_sum / result_sum
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """
        Run comprehensive benchmark across all test cases and sizes.
        
        Returns:
            DataFrame with all benchmark results
        """
        test_cases = [
            'alternating_large',
            'geometric_series', 
            'harmonic_series',
            'mixed_magnitude',
            'pathological_cancellation',
            'random_normal',
            'random_exponential',
            'ill_conditioned'
        ]
        
        sizes = [100, 1000, 10000, 100000]
        dtypes = [np.float32, np.float64]
        
        print("Running comprehensive accuracy benchmark...")
        print(f"Test cases: {len(test_cases)}")
        print(f"Sizes: {sizes}")
        print(f"Data types: {[dt.__name__ for dt in dtypes]}")
        print(f"Total combinations: {len(test_cases) * len(sizes) * len(dtypes)}")
        print()
        
        total_tests = len(test_cases) * len(sizes) * len(dtypes)
        test_count = 0
        
        for case_type in test_cases:
            for size in sizes:
                for dtype in dtypes:
                    test_count += 1
                    test_name = f"{case_type}_{dtype.__name__}_{size}"
                    
                    print(f"[{test_count}/{total_tests}] Running {test_name}...")
                    
                    try:
                        data, exact = self.generate_test_case(case_type, size, dtype)
                        result = self.run_single_benchmark(test_name, data, exact)
                        result['case_type'] = case_type
                        result['dtype'] = dtype.__name__
                        self.results.append(result)
                        
                    except Exception as e:
                        print(f"  Error: {e}")
                        continue
        
        return pd.DataFrame(self.results)
    
    def analyze_results(self, df: pd.DataFrame) -> None:
        """
        Analyze and display benchmark results.
        
        Args:
            df: DataFrame with benchmark results
        """
        print("\n" + "="*80)
        print("ACCURACY BENCHMARK ANALYSIS")
        print("="*80)
        
        # Algorithm comparison by error
        algorithms = ['numpy', 'kahan', 'einstein', 'tree', 'parallel']
        
        print("\nAVERAGE RELATIVE ERROR BY ALGORITHM:")
        print("-" * 50)
        print(f"{'Algorithm':<12} {'Mean Rel Error':<15} {'Median Rel Error':<15} {'Max Rel Error':<15}")
        print("-" * 60)
        
        for alg in algorithms:
            col = f'{alg}_rel_error'
            if col in df.columns:
                mean_error = df[col].mean()
                median_error = df[col].median()
                max_error = df[col].max()
                print(f"{alg:<12} {mean_error:<15.2e} {median_error:<15.2e} {max_error:<15.2e}")
        
        # Error by test case type
        print("\nERROR BY TEST CASE TYPE:")
        print("-" * 40)
        
        for case_type in df['case_type'].unique():
            case_df = df[df['case_type'] == case_type]
            print(f"\n{case_type}:")
            
            for alg in algorithms:
                col = f'{alg}_rel_error'
                if col in case_df.columns:
                    median_error = case_df[col].median()
                    print(f"  {alg}: {median_error:.2e}")
        
        # Performance comparison
        print("\nPERFORMANCE COMPARISON:")
        print("-" * 30)
        print(f"{'Algorithm':<12} {'Mean Time (ms)':<15} {'Relative Speed':<15}")
        print("-" * 45)
        
        numpy_time = df['numpy_time'].mean()
        
        for alg in algorithms:
            col = f'{alg}_time'
            if col in df.columns:
                mean_time = df[col].mean() * 1000  # Convert to ms
                relative_speed = numpy_time / df[col].mean()
                print(f"{alg:<12} {mean_time:<15.3f} {relative_speed:<15.2f}x")
        
        # Condition number analysis
        print("\nCONDITION NUMBER ANALYSIS:")
        print("-" * 30)
        
        # Bin by condition number
        df['condition_bin'] = pd.cut(df['condition_number'], 
                                   bins=[1, 10, 100, 1000, 10000, np.inf],
                                   labels=['1-10', '10-100', '100-1K', '1K-10K', '>10K'])
        
        for bin_name in df['condition_bin'].cat.categories:
            bin_df = df[df['condition_bin'] == bin_name]
            if len(bin_df) > 0:
                print(f"\nCondition number {bin_name}:")
                for alg in algorithms:
                    col = f'{alg}_rel_error'
                    if col in bin_df.columns:
                        median_error = bin_df[col].median()
                        print(f"  {alg}: {median_error:.2e}")
    
    def plot_results(self, df: pd.DataFrame, save_plots: bool = True) -> None:
        """
        Create visualization plots of benchmark results.
        
        Args:
            df: DataFrame with benchmark results
            save_plots: Whether to save plots to files
        """
        algorithms = ['numpy', 'kahan', 'einstein', 'tree', 'parallel']
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Plot 1: Error vs Size
        plt.figure(figsize=(12, 8))
        
        for i, alg in enumerate(algorithms):
            col = f'{alg}_rel_error'
            if col in df.columns:
                # Group by size and calculate median error
                size_errors = df.groupby('size')[col].median()
                plt.loglog(size_errors.index, size_errors.values, 
                          'o-', color=colors[i], label=alg, markersize=6)
        
        plt.xlabel('Array Size')
        plt.ylabel('Median Relative Error')
        plt.title('Accuracy vs Array Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('accuracy_vs_size.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Error by Test Case
        plt.figure(figsize=(14, 8))
        
        case_types = df['case_type'].unique()
        x_pos = np.arange(len(case_types))
        
        bar_width = 0.15
        
        for i, alg in enumerate(algorithms):
            col = f'{alg}_rel_error'
            if col in df.columns:
                case_errors = [df[df['case_type'] == case][col].median() 
                              for case in case_types]
                plt.bar(x_pos + i * bar_width, case_errors, bar_width, 
                       color=colors[i], label=alg, alpha=0.8)
        
        plt.xlabel('Test Case Type')
        plt.ylabel('Median Relative Error (log scale)')
        plt.title('Accuracy by Test Case Type')
        plt.yscale('log')
        plt.xticks(x_pos + bar_width * 2, case_types, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('accuracy_by_test_case.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Performance vs Accuracy Trade-off
        plt.figure(figsize=(10, 8))
        
        for i, alg in enumerate(algorithms):
            time_col = f'{alg}_time'
            error_col = f'{alg}_rel_error'
            
            if time_col in df.columns and error_col in df.columns:
                mean_time = df[time_col].mean() * 1000  # Convert to ms
                median_error = df[error_col].median()
                
                plt.scatter(mean_time, median_error, s=100, color=colors[i], 
                           label=alg, alpha=0.8)
                plt.annotate(alg, (mean_time, median_error), 
                           xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Mean Execution Time (ms)')
        plt.ylabel('Median Relative Error')
        plt.title('Performance vs Accuracy Trade-off')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('performance_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run the complete accuracy benchmark suite."""
    print("KAHAN SUMMATION LIBRARY - ACCURACY BENCHMARK")
    print("=" * 60)
    
    # Create and run benchmark
    benchmark = AccuracyBenchmark()
    results_df = benchmark.run_comprehensive_benchmark()
    
    # Save results
    results_df.to_csv('accuracy_benchmark_results.csv', index=False)
    print(f"\nResults saved to accuracy_benchmark_results.csv")
    
    # Analyze results
    benchmark.analyze_results(results_df)
    
    # Create plots
    try:
        benchmark.plot_results(results_df)
    except ImportError:
        print("\nMatplotlib not available, skipping plots")
    except Exception as e:
        print(f"\nError creating plots: {e}")
    
    print("\n" + "="*60)
    print("Accuracy benchmark completed!")
    print("="*60)


if __name__ == "__main__":
    main()