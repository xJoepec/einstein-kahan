"""
Test suite for Kahan Summation Library.

This package contains comprehensive tests for all components of the
Kahan summation library, including unit tests, integration tests,
and performance benchmarks.

Test Structure:
- test_core.py: Tests for core classes and functions
- test_algorithms.py: Tests for high-level algorithms  
- test_integration.py: Integration and end-to-end tests
- test_performance.py: Performance and benchmark tests
- conftest.py: Shared fixtures and configuration

Usage:
    # Run all tests
    pytest

    # Run specific test file
    pytest tests/test_core.py

    # Run tests with coverage
    pytest --cov=kahan

    # Run only fast tests
    pytest -m "not slow"

    # Run only C++ tests (if extensions available)
    pytest -m cpp
"""

__version__ = "1.0.0"