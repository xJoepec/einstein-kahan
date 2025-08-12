#!/usr/bin/env python3
"""
Pytest configuration and fixtures for Kahan summation tests.

This file contains shared test fixtures, configuration, and utilities
used across the test suite.
"""

import pytest
import numpy as np
import torch
from typing import List, Tuple, Generator
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def simple_data():
    """Simple test data for basic functionality tests."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def challenging_float32():
    """Challenging float32 data that exposes precision issues."""
    return np.array([1e8, 1.0, -1e8], dtype=np.float32)


@pytest.fixture
def large_mixed_data():
    """Large dataset with mixed magnitudes for stress testing."""
    np.random.seed(42)
    n = 10000
    
    # Mix of different magnitude ranges
    large_positive = np.random.normal(1e6, 1e5, n//4)
    large_negative = np.random.normal(-1e6, 1e5, n//4)
    small_positive = np.random.normal(0, 1, n//4)
    small_negative = np.random.normal(0, 1, n//4)
    
    data = np.concatenate([large_positive, large_negative, small_positive, small_negative])
    np.random.shuffle(data)
    
    return data.astype(np.float32)


@pytest.fixture
def pathological_cancellation():
    """Data designed to cause maximum cancellation errors."""
    n = 1000
    epsilon = np.finfo(np.float32).eps * 10
    
    data = np.zeros(n, dtype=np.float32)
    data[::2] = 1.0
    data[1::2] = -1.0 + epsilon
    
    return data


@pytest.fixture
def geometric_series_data():
    """Geometric series data for convergence testing."""
    n = 100
    return (0.5 ** np.arange(n)).astype(np.float32)


@pytest.fixture
def harmonic_series_data():
    """Harmonic series data for accuracy testing."""
    n = 1000
    return (1.0 / np.arange(1, n + 1)).astype(np.float32)


@pytest.fixture
def random_normal_data():
    """Random normal distribution data."""
    np.random.seed(42)
    return np.random.normal(0, 1, 10000).astype(np.float32)


@pytest.fixture
def random_exponential_data():
    """Random exponential distribution data."""
    np.random.seed(42)
    return np.random.exponential(1.0, 10000).astype(np.float32)


@pytest.fixture
def ill_conditioned_data():
    """Ill-conditioned data spanning many orders of magnitude."""
    np.random.seed(42)
    n = 1000
    
    # Generate numbers spanning many orders of magnitude
    exponents = np.random.uniform(-10, 10, n)
    signs = np.random.choice([-1, 1], n)
    data = (signs * 10.0 ** exponents).astype(np.float32)
    
    return data


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request):
    """Parameterized fixture for different data types."""
    return request.param


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    """Parameterized fixture for different devices."""
    return torch.device(request.param)


@pytest.fixture
def batch_arrays():
    """Collection of arrays for batch processing tests."""
    np.random.seed(42)
    
    arrays = [
        np.random.randn(100).astype(np.float32),
        np.random.randn(200).astype(np.float32),
        np.random.randn(150).astype(np.float32),
        np.random.randn(300).astype(np.float32),
        np.random.randn(80).astype(np.float32),
    ]
    
    return arrays


class AccuracyChecker:
    """Utility class for checking numerical accuracy."""
    
    @staticmethod
    def relative_error(computed: float, reference: float) -> float:
        """Calculate relative error."""
        if reference == 0:
            return abs(computed)
        return abs(computed - reference) / abs(reference)
    
    @staticmethod
    def high_precision_sum(values) -> float:
        """Compute sum in high precision for reference."""
        if hasattr(values, 'astype'):
            return float(np.sum(values.astype(np.float64)))
        else:
            return float(np.sum(np.array(values, dtype=np.float64)))
    
    @staticmethod
    def condition_number(values) -> float:
        """Estimate condition number for summation."""
        if hasattr(values, 'astype'):
            values_array = values.astype(np.float64)
        else:
            values_array = np.array(values, dtype=np.float64)
        
        abs_sum = np.sum(np.abs(values_array))
        result_sum = abs(np.sum(values_array))
        
        if result_sum == 0:
            return float('inf')
        return abs_sum / result_sum


@pytest.fixture
def accuracy_checker():
    """Fixture providing accuracy checking utilities."""
    return AccuracyChecker()


class PerformanceTimer:
    """Utility class for performance timing."""
    
    def __init__(self):
        self.times = []
    
    def time_function(self, func, *args, num_runs=5, **kwargs):
        """Time a function execution."""
        import time
        
        times = []
        
        # Warm-up run
        func(*args, **kwargs)
        
        # Timed runs
        for _ in range(num_runs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'result': result
        }


@pytest.fixture
def performance_timer():
    """Fixture providing performance timing utilities."""
    return PerformanceTimer()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cpp: marks tests that require C++ extensions"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests that take long time as slow
        if "large" in item.name or "stress" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark tests that require C++ extensions
        if "cpp" in item.name or "c++" in item.name:
            item.add_marker(pytest.mark.cpp)
        
        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def test_data_generator():
    """Generator for various test data patterns."""
    
    class TestDataGenerator:
        def __init__(self):
            self.seed = 42
        
        def alternating_large_small(self, size: int, dtype=np.float32):
            """Generate alternating large and small values."""
            data = np.zeros(size, dtype=dtype)
            data[::2] = 1e8
            data[1::2] = 1.0
            return data
        
        def geometric_progression(self, size: int, ratio: float = 0.5, dtype=np.float32):
            """Generate geometric progression."""
            return (ratio ** np.arange(size)).astype(dtype)
        
        def mixed_magnitudes(self, size: int, dtype=np.float32):
            """Generate data with mixed magnitudes."""
            np.random.seed(self.seed)
            
            # Different magnitude ranges
            n_quarter = size // 4
            
            large_vals = np.random.normal(1e6, 1e5, n_quarter)
            medium_vals = np.random.normal(1e3, 1e2, n_quarter)
            small_vals = np.random.normal(1, 0.1, n_quarter)
            tiny_vals = np.random.normal(0, 1e-6, size - 3 * n_quarter)
            
            data = np.concatenate([large_vals, medium_vals, small_vals, tiny_vals])
            np.random.shuffle(data)
            
            return data.astype(dtype)
        
        def ill_conditioned(self, size: int, dtype=np.float32):
            """Generate ill-conditioned summation problem."""
            np.random.seed(self.seed)
            
            # Random exponents spanning wide range
            exponents = np.random.uniform(-15, 15, size)
            signs = np.random.choice([-1, 1], size)
            
            return (signs * 10.0 ** exponents).astype(dtype)
        
        def pathological_cancellation(self, size: int, dtype=np.float32):
            """Generate data designed for maximum cancellation errors."""
            epsilon = np.finfo(dtype).eps * 10
            
            data = np.zeros(size, dtype=dtype)
            data[::2] = 1.0
            data[1::2] = -1.0 + epsilon
            
            return data
    
    return TestDataGenerator()


@pytest.fixture
def error_tolerance():
    """Fixture providing error tolerances for different precisions."""
    return {
        np.float16: 1e-3,
        np.float32: 1e-6,
        np.float64: 1e-14,
        torch.float16: 1e-3,
        torch.float32: 1e-6,
        torch.float64: 1e-14,
    }


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )


def skip_if_no_cpp():
    """Skip test if C++ extensions are not available."""
    try:
        import kahan.kahan_cpp
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skipif(True, reason="C++ extensions not available")


# Custom assertion helpers
def assert_arrays_close(a, b, rtol=1e-7, atol=1e-14):
    """Assert that two arrays are close with informative error messages."""
    if hasattr(a, 'numpy'):
        a = a.numpy()
    if hasattr(b, 'numpy'):
        b = b.numpy()
    
    a = np.asarray(a)
    b = np.asarray(b)
    
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.abs(a - b)
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        max_diff = diff[max_diff_idx]
        
        raise AssertionError(
            f"Arrays not close enough:\n"
            f"Max difference: {max_diff} at index {max_diff_idx}\n"
            f"Values: {a[max_diff_idx]} vs {b[max_diff_idx]}\n"
            f"Relative tolerance: {rtol}, Absolute tolerance: {atol}"
        )


def assert_relative_error(computed, reference, max_relative_error):
    """Assert that relative error is within bounds."""
    if reference == 0:
        assert abs(computed) <= max_relative_error
    else:
        relative_error = abs(computed - reference) / abs(reference)
        assert relative_error <= max_relative_error, (
            f"Relative error {relative_error} exceeds threshold {max_relative_error}\n"
            f"Computed: {computed}, Reference: {reference}"
        )