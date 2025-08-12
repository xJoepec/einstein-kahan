#!/usr/bin/env python3
"""
Unit tests for core Kahan summation functionality.

Tests the fundamental classes and functions in the kahan.core module.
"""

import pytest
import numpy as np
import torch
import math
from typing import List
import sys
sys.path.append('..')

from kahan.core import (
    KahanAccumulator,
    EinsteinKahanSummation,
    kahan_add,
    compensated_dot_product,
    RiemannCurvatureTensor
)


class TestKahanAccumulator:
    """Test cases for KahanAccumulator class."""
    
    def test_basic_functionality(self):
        """Test basic accumulator operations."""
        acc = KahanAccumulator()
        
        # Test initial state
        assert acc.get() == 0.0
        assert acc.c.item() == 0.0
        
        # Test single addition
        acc.add(1.0)
        assert acc.get() == 1.0
        
        # Test multiple additions
        acc.add(2.0)
        acc.add(3.0)
        assert acc.get() == 6.0
    
    def test_precision_improvement(self):
        """Test that Kahan accumulator improves precision."""
        # Create scenario where naive summation loses precision
        values = [1e8, 1.0, -1e8]
        
        # Naive summation (loses precision in float32)
        naive_sum = 0.0
        for v in values:
            naive_sum = np.float32(naive_sum + v)
        
        # Kahan summation
        acc = KahanAccumulator(dtype=torch.float32)
        for v in values:
            acc.add(v)
        kahan_result = acc.get().item()
        
        # Kahan should be more accurate
        expected = 1.0
        assert abs(kahan_result - expected) < abs(naive_sum - expected)
    
    def test_tensor_shapes(self):
        """Test accumulator with different tensor shapes."""
        shapes = [(), (3,), (2, 3), (2, 3, 4)]
        
        for shape in shapes:
            acc = KahanAccumulator(shape=shape)
            
            # Add values
            value = torch.ones(shape)
            acc.add(value)
            acc.add(2 * value)
            
            result = acc.get()
            expected = 3 * torch.ones(shape)
            
            assert result.shape == shape
            assert torch.allclose(result, expected)
    
    def test_different_dtypes(self):
        """Test accumulator with different data types."""
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            acc = KahanAccumulator(dtype=dtype)
            
            acc.add(1.5)
            acc.add(2.7)
            
            result = acc.get()
            assert result.dtype == dtype
            assert abs(result.item() - 4.2) < 1e-6
    
    def test_reset(self):
        """Test accumulator reset functionality."""
        acc = KahanAccumulator()
        
        acc.add(5.0)
        acc.add(3.0)
        assert acc.get() == 8.0
        
        acc.reset()
        assert acc.get() == 0.0
        assert acc.c.item() == 0.0
    
    def test_device_handling(self):
        """Test accumulator on different devices."""
        devices = [torch.device('cpu')]
        
        # Add GPU if available
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        
        for device in devices:
            acc = KahanAccumulator(device=device)
            
            value = torch.tensor(2.5, device=device)
            acc.add(value)
            
            result = acc.get()
            assert result.device == device
            assert result.item() == 2.5


class TestEinsteinKahanSummation:
    """Test cases for EinsteinKahanSummation class."""
    
    def test_initialization(self):
        """Test module initialization."""
        dim = 64
        module = EinsteinKahanSummation(dim)
        
        assert module.dim == dim
        assert module.einstein_constant == 8 * math.pi
        assert module.get_total_error() == 0.0
        assert module.get_operation_count() == 0
    
    def test_custom_einstein_constant(self):
        """Test custom Einstein constant."""
        custom_constant = 4 * math.pi
        module = EinsteinKahanSummation(64, einstein_constant=custom_constant)
        
        assert module.einstein_constant == custom_constant
    
    def test_basic_einsum(self):
        """Test basic Einstein summation operation."""
        dim = 32
        module = EinsteinKahanSummation(dim)
        
        # Matrix multiplication: (2, 32) × (32, 16) -> (2, 16)
        a = torch.randn(2, 32)
        b = torch.randn(32, 16)
        
        result = module(a, b, "ij,jk->ik")
        
        assert result.shape == (2, 16)
        assert module.get_operation_count() == 1
        
        # Compare with standard einsum (should be close)
        expected = torch.einsum("ij,jk->ik", a, b)
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_error_tracking(self):
        """Test error accumulation tracking."""
        module = EinsteinKahanSummation(16)
        
        # Perform operations that accumulate error
        for _ in range(5):
            a = torch.randn(4, 16, dtype=torch.float32)
            b = torch.randn(16, 8, dtype=torch.float32)
            _ = module(a, b)
        
        assert module.get_operation_count() == 5
        # Error should be tracked (non-zero for float32)
        assert module.get_total_error() >= 0
    
    def test_different_equations(self):
        """Test various Einstein summation equations."""
        module = EinsteinKahanSummation(8)
        
        test_cases = [
            # Dot product
            ("i,i->", torch.randn(8), torch.randn(8)),
            # Matrix multiplication
            ("ij,jk->ik", torch.randn(4, 8), torch.randn(8, 6)),
            # Batch matrix multiplication
            ("bij,bjk->bik", torch.randn(3, 4, 8), torch.randn(3, 8, 6)),
            # Element-wise multiplication and sum
            ("ij,ij->", torch.randn(4, 8), torch.randn(4, 8)),
        ]
        
        for equation, tensor1, tensor2 in test_cases:
            result = module(tensor1, tensor2, equation)
            expected = torch.einsum(equation, tensor1, tensor2)
            
            assert torch.allclose(result, expected, atol=1e-4)
    
    def test_reset_error_tracking(self):
        """Test error tracking reset."""
        module = EinsteinKahanSummation(16)
        
        # Perform some operations
        a = torch.randn(4, 16)
        b = torch.randn(16, 8)
        _ = module(a, b)
        
        assert module.get_operation_count() > 0
        
        # Reset tracking
        module.reset_error_tracking()
        
        assert module.get_operation_count() == 0
        assert module.get_total_error() == 0.0


class TestKahanAdd:
    """Test cases for kahan_add function."""
    
    def test_basic_addition(self):
        """Test basic Kahan addition step."""
        a = 1.0
        b = 2.0
        c = 0.0
        
        new_sum, new_comp = kahan_add(a, b, c)
        
        assert new_sum == 3.0
        assert new_comp == 0.0  # No error in this simple case
    
    def test_error_compensation(self):
        """Test error compensation in difficult case."""
        # Case where standard addition loses precision
        a = np.float32(1e8)
        b = np.float32(1.0)
        c = 0.0
        
        # Standard addition
        naive_result = np.float32(a + b)
        
        # Kahan addition
        kahan_result, compensation = kahan_add(float(a), float(b), c)
        
        # Kahan should capture the lost precision in compensation
        assert compensation != 0.0
        
        # The compensated result should be more accurate
        true_result = float(a) + float(b)
        kahan_corrected = kahan_result + compensation
        
        assert abs(kahan_corrected - true_result) < abs(naive_result - true_result)
    
    def test_multiple_steps(self):
        """Test multiple Kahan addition steps."""
        values = [1e8, 1.0, 2.0, -1e8, 3.0]
        
        sum_val = 0.0
        comp = 0.0
        
        for value in values:
            sum_val, comp = kahan_add(sum_val, value, comp)
        
        # Final result should be close to expected
        expected = sum(values)
        assert abs(sum_val - expected) < 1e-10


class TestCompensatedDotProduct:
    """Test cases for compensated_dot_product function."""
    
    def test_basic_dot_product(self):
        """Test basic compensated dot product."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        
        result, error = compensated_dot_product(a, b)
        expected = torch.dot(a, b)
        
        assert torch.allclose(result, expected)
        assert isinstance(error, torch.Tensor)
    
    def test_precision_improvement(self):
        """Test that compensated dot product improves precision."""
        # Create vectors that stress floating-point precision
        n = 1000
        a = torch.ones(n, dtype=torch.float32) * 1e4
        b = torch.ones(n, dtype=torch.float32) * 1e-4
        
        # Standard dot product
        standard_result = torch.dot(a, b)
        
        # Compensated dot product
        compensated_result, error = compensated_dot_product(a, b)
        
        # Expected result (in higher precision)
        expected = float(n)  # 1000 * 1e4 * 1e-4 = 1000
        
        # Compensated should be more accurate
        standard_error = abs(standard_result.item() - expected)
        compensated_error = abs(compensated_result.item() - expected)
        
        assert compensated_error <= standard_error
    
    def test_shape_mismatch(self):
        """Test error handling for mismatched shapes."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        
        with pytest.raises(AssertionError):
            compensated_dot_product(a, b)
    
    def test_different_dtypes(self):
        """Test with different data types."""
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            a = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
            b = torch.tensor([4.0, 5.0, 6.0], dtype=dtype)
            
            result, error = compensated_dot_product(a, b)
            
            assert result.dtype == dtype
            assert torch.allclose(result, torch.tensor(32.0, dtype=dtype))


class TestRiemannCurvatureTensor:
    """Test cases for RiemannCurvatureTensor class."""
    
    def test_initialization(self):
        """Test tensor initialization."""
        dim = 8
        tensor = RiemannCurvatureTensor(dim)
        
        assert tensor.dim == dim
        assert tensor.n_components > 0
        assert len(tensor.components) == tensor.n_components
    
    def test_symmetries(self):
        """Test Riemann tensor symmetries."""
        dim = 4
        tensor = RiemannCurvatureTensor(dim)
        
        # Test antisymmetry: R_ijkl = -R_jikl = -R_ijlk
        i, j, k, l = 0, 1, 2, 3
        
        R_ijkl = tensor.get_component(i, j, k, l)
        R_jikl = tensor.get_component(j, i, k, l)
        R_ijlk = tensor.get_component(i, j, l, k)
        
        assert torch.allclose(R_ijkl, -R_jikl, atol=1e-6)
        assert torch.allclose(R_ijkl, -R_ijlk, atol=1e-6)
        
        # Test symmetry: R_ijkl = R_klij
        R_klij = tensor.get_component(k, l, i, j)
        assert torch.allclose(R_ijkl, R_klij, atol=1e-6)
    
    def test_ricci_computation(self):
        """Test Ricci tensor computation."""
        dim = 4
        tensor = RiemannCurvatureTensor(dim)
        
        ricci = tensor.compute_ricci()
        
        assert ricci.shape == (dim, dim)
        assert ricci.dtype == tensor.components.dtype
    
    def test_scalar_curvature(self):
        """Test scalar curvature computation."""
        dim = 4
        tensor = RiemannCurvatureTensor(dim)
        
        scalar = tensor.compute_scalar_curvature()
        
        assert scalar.ndim == 0  # Should be a scalar
        assert scalar.dtype == tensor.components.dtype
    
    def test_forward_pass(self):
        """Test forward pass."""
        dim = 4
        tensor = RiemannCurvatureTensor(dim)
        
        ricci, scalar = tensor.forward()
        
        assert ricci.shape == (dim, dim)
        assert scalar.ndim == 0
    
    def test_large_dimension_handling(self):
        """Test handling of large dimensions."""
        # Test that large dimensions are capped appropriately
        large_dim = 100
        tensor = RiemannCurvatureTensor(large_dim)
        
        # Should cap the actual computation dimension
        assert tensor.n_components <= 10000  # As per implementation
        
        # Should still work
        ricci, scalar = tensor.forward()
        assert ricci.shape == (large_dim, large_dim)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_precision_consistency(dtype):
    """Test that algorithms work consistently across precisions."""
    # Test data
    data = torch.tensor([1e6, 1.0, 2.0, -1e6, 3.0], dtype=dtype)
    
    # KahanAccumulator
    acc = KahanAccumulator(dtype=dtype)
    for value in data:
        acc.add(value)
    
    result = acc.get()
    expected = torch.sum(data)
    
    assert result.dtype == dtype
    assert torch.allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_device_consistency(device):
    """Test that algorithms work on different devices."""
    device = torch.device(device)
    
    # Test KahanAccumulator on device
    acc = KahanAccumulator(device=device)
    value = torch.tensor(3.14, device=device)
    
    acc.add(value)
    result = acc.get()
    
    assert result.device == device
    assert torch.allclose(result, value)


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Empty accumulator
    acc = KahanAccumulator()
    assert acc.get() == 0.0
    
    # Very small values
    acc.add(1e-100)
    acc.add(1e-100)
    assert acc.get() > 0
    
    # Very large values
    acc.reset()
    acc.add(1e100)
    acc.add(-1e100)
    assert abs(acc.get()) < 1e90  # Should be much smaller due to cancellation
    
    # NaN handling
    acc.reset()
    acc.add(float('nan'))
    assert torch.isnan(acc.get())
    
    # Infinity handling
    acc.reset()
    acc.add(float('inf'))
    assert torch.isinf(acc.get())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])