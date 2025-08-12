#!/usr/bin/env python3
"""
Scientific computing examples using Kahan summation.

Demonstrates applications in physics simulations, numerical analysis,
and scientific computing where precision matters.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List
import sys
sys.path.append('..')

from kahan import (
    kahan_sum,
    einstein_kahan_sum,
    tree_reduce_kahan,
    compensated_mean,
    compensated_variance,
    KahanAccumulator
)


def particle_simulation_example():
    """
    N-body particle simulation with precise center of mass calculation.
    
    In particle physics simulations, maintaining conservation laws
    (like center of mass) requires high numerical precision.
    """
    print("=" * 70)
    print("SCIENTIFIC EXAMPLE: N-Body Particle Simulation")
    print("=" * 70)
    
    # Simulation parameters
    n_particles = 100000
    dimension = 3
    
    # Generate random particle positions and masses
    np.random.seed(42)
    positions = np.random.uniform(-1000, 1000, (n_particles, dimension)).astype(np.float32)
    masses = np.random.exponential(1.0, n_particles).astype(np.float32)
    
    print(f"Simulating {n_particles} particles in {dimension}D")
    print(f"Position range: [{positions.min():.1f}, {positions.max():.1f}]")
    print(f"Mass range: [{masses.min():.3f}, {masses.max():.3f}]")
    print()
    
    # Calculate center of mass using different methods
    def calculate_center_of_mass_naive(pos, mass):
        """Standard floating-point calculation."""
        total_mass = np.sum(mass)
        com = np.sum(pos * mass[:, np.newaxis], axis=0) / total_mass
        return com, total_mass
    
    def calculate_center_of_mass_kahan(pos, mass):
        """Using Kahan summation for precision."""
        total_mass = kahan_sum(mass)
        
        com = np.zeros(dimension)
        for d in range(dimension):
            weighted_positions = pos[:, d] * mass
            com[d] = kahan_sum(weighted_positions) / total_mass
        
        return com, total_mass
    
    def calculate_center_of_mass_einstein(pos, mass):
        """Using Einstein-Kahan summation."""
        total_mass, mass_curvature = einstein_kahan_sum(mass)
        
        com = np.zeros(dimension)
        curvatures = np.zeros(dimension)
        
        for d in range(dimension):
            weighted_positions = pos[:, d] * mass
            com[d], curvatures[d] = einstein_kahan_sum(weighted_positions)
            com[d] /= total_mass
        
        return com, total_mass, mass_curvature, curvatures
    
    # Compare methods
    print("Center of Mass Calculation:")
    print("-" * 40)
    
    # Naive method
    start_time = time.time()
    com_naive, mass_naive = calculate_center_of_mass_naive(positions, masses)
    time_naive = (time.time() - start_time) * 1000
    
    # Kahan method
    start_time = time.time()
    com_kahan, mass_kahan = calculate_center_of_mass_kahan(positions, masses)
    time_kahan = (time.time() - start_time) * 1000
    
    # Einstein-Kahan method
    start_time = time.time()
    com_einstein, mass_einstein, mass_curv, pos_curv = calculate_center_of_mass_einstein(positions, masses)
    time_einstein = (time.time() - start_time) * 1000
    
    print(f"{'Method':<15} {'Time (ms)':<10} {'Total Mass':<15} {'COM X':<12} {'COM Y':<12} {'COM Z':<12}")
    print("-" * 85)
    print(f"{'Naive':<15} {time_naive:<10.2f} {mass_naive:<15.6f} {com_naive[0]:<12.6f} {com_naive[1]:<12.6f} {com_naive[2]:<12.6f}")
    print(f"{'Kahan':<15} {time_kahan:<10.2f} {mass_kahan:<15.6f} {com_kahan[0]:<12.6f} {com_kahan[1]:<12.6f} {com_kahan[2]:<12.6f}")
    print(f"{'Einstein':<15} {time_einstein:<10.2f} {mass_einstein:<15.6f} {com_einstein[0]:<12.6f} {com_einstein[1]:<12.6f} {com_einstein[2]:<12.6f}")
    print()
    
    # Show error analysis
    print("Error Analysis:")
    print("-" * 40)
    print(f"Mass difference (Kahan vs Naive):    {abs(mass_kahan - mass_naive):.2e}")
    print(f"Mass difference (Einstein vs Naive): {abs(mass_einstein - mass_naive):.2e}")
    print(f"COM difference (Kahan vs Naive):     {np.linalg.norm(com_kahan - com_naive):.2e}")
    print(f"COM difference (Einstein vs Naive):  {np.linalg.norm(com_einstein - com_naive):.2e}")
    print()
    
    # Einstein-Kahan specific analysis
    print("Einstein-Kahan Curvature Analysis:")
    print("-" * 40)
    print(f"Mass curvature error:     {mass_curv:.2e}")
    print(f"Position curvature (X):   {pos_curv[0]:.2e}")
    print(f"Position curvature (Y):   {pos_curv[1]:.2e}")
    print(f"Position curvature (Z):   {pos_curv[2]:.2e}")
    print()


def monte_carlo_integration():
    """
    Monte Carlo integration with error compensation.
    
    Demonstrates how Kahan summation improves accuracy in
    statistical sampling methods.
    """
    print("=" * 70)
    print("SCIENTIFIC EXAMPLE: Monte Carlo Integration")
    print("=" * 70)
    
    def integrand(x):
        """Function to integrate: sin(x) * exp(-x^2)"""
        return np.sin(x) * np.exp(-x**2)
    
    def true_integral():
        """Analytical result for comparison (approximate)."""
        # This is a known integral result
        return 0.62049958742306  # Computed with high precision
    
    # Integration parameters
    a, b = -3, 3  # Integration bounds
    n_samples_list = [1000, 10000, 100000, 1000000]
    
    print(f"Integrating sin(x) * exp(-x^2) from {a} to {b}")
    print(f"True value: {true_integral():.10f}")
    print()
    
    print(f"{'N Samples':<12} {'Naive MC':<15} {'Kahan MC':<15} {'Einstein MC':<15} {'Naive Error':<12} {'Kahan Error':<12} {'Einstein Error':<15}")
    print("-" * 110)
    
    for n_samples in n_samples_list:
        np.random.seed(42)  # Reproducible results
        
        # Generate random samples
        x_samples = np.random.uniform(a, b, n_samples).astype(np.float32)
        y_samples = integrand(x_samples)
        
        # Scale factor for integration
        scale = (b - a) / n_samples
        
        # Monte Carlo integration with different summation methods
        naive_sum = np.sum(y_samples) * scale
        kahan_sum_result = kahan_sum(y_samples) * scale
        einstein_sum_result, _ = einstein_kahan_sum(y_samples)
        einstein_sum_result *= scale
        
        # Calculate errors
        true_val = true_integral()
        naive_error = abs(naive_sum - true_val)
        kahan_error = abs(kahan_sum_result - true_val)
        einstein_error = abs(einstein_sum_result - true_val)
        
        print(f"{n_samples:<12} {naive_sum:<15.8f} {kahan_sum_result:<15.8f} {einstein_sum_result:<15.8f} "
              f"{naive_error:<12.2e} {kahan_error:<12.2e} {einstein_error:<15.2e}")
    
    print()


def signal_processing_example():
    """
    Signal processing with noise reduction using compensated summation.
    
    Shows how precision affects signal-to-noise ratio calculations.
    """
    print("=" * 70)
    print("SCIENTIFIC EXAMPLE: Signal Processing")
    print("=" * 70)
    
    # Generate test signal
    t = np.linspace(0, 1, 10000, dtype=np.float32)
    
    # Clean signal: sum of sinusoids
    signal_clean = (np.sin(2 * np.pi * 5 * t) + 
                   0.5 * np.sin(2 * np.pi * 10 * t) + 
                   0.25 * np.sin(2 * np.pi * 20 * t))
    
    # Add noise
    np.random.seed(123)
    noise = np.random.normal(0, 0.1, len(t)).astype(np.float32)
    signal_noisy = signal_clean + noise
    
    print(f"Signal length: {len(signal_noisy)}")
    print(f"Sample rate: {len(t)} Hz")
    print(f"Noise level: 0.1 RMS")
    print()
    
    def calculate_power(signal):
        """Calculate signal power using different methods."""
        # Power = mean of squared signal
        signal_squared = signal ** 2
        
        naive_power = np.mean(signal_squared)
        kahan_power = compensated_mean(signal_squared)
        einstein_power, curvature = einstein_kahan_sum(signal_squared)
        einstein_power /= len(signal_squared)
        
        return naive_power, kahan_power, einstein_power, curvature
    
    def calculate_snr(signal_power, noise_power):
        """Calculate signal-to-noise ratio in dB."""
        return 10 * np.log10(signal_power / noise_power)
    
    # Calculate power for clean signal, noise, and noisy signal
    print("Power Analysis:")
    print("-" * 50)
    
    clean_powers = calculate_power(signal_clean)
    noise_powers = calculate_power(noise)
    noisy_powers = calculate_power(signal_noisy)
    
    print(f"{'Component':<12} {'Naive':<12} {'Kahan':<12} {'Einstein':<12} {'Curvature':<12}")
    print("-" * 65)
    print(f"{'Clean Signal':<12} {clean_powers[0]:<12.6f} {clean_powers[1]:<12.6f} {clean_powers[2]:<12.6f} {clean_powers[3]:<12.2e}")
    print(f"{'Noise':<12} {noise_powers[0]:<12.6f} {noise_powers[1]:<12.6f} {noise_powers[2]:<12.6f} {noise_powers[3]:<12.2e}")
    print(f"{'Noisy Signal':<12} {noisy_powers[0]:<12.6f} {noisy_powers[1]:<12.6f} {noisy_powers[2]:<12.6f} {noisy_powers[3]:<12.2e}")
    print()
    
    # Calculate SNR
    print("Signal-to-Noise Ratio:")
    print("-" * 30)
    
    snr_naive = calculate_snr(clean_powers[0], noise_powers[0])
    snr_kahan = calculate_snr(clean_powers[1], noise_powers[1])
    snr_einstein = calculate_snr(clean_powers[2], noise_powers[2])
    
    print(f"Naive SNR:     {snr_naive:.2f} dB")
    print(f"Kahan SNR:     {snr_kahan:.2f} dB")
    print(f"Einstein SNR:  {snr_einstein:.2f} dB")
    print()


def numerical_differentiation():
    """
    Numerical differentiation with error compensation.
    
    Shows how floating-point errors accumulate in finite difference
    calculations and how Kahan summation helps.
    """
    print("=" * 70)
    print("SCIENTIFIC EXAMPLE: Numerical Differentiation")
    print("=" * 70)
    
    def test_function(x):
        """Test function: f(x) = x^3 * sin(x)"""
        return x**3 * np.sin(x)
    
    def true_derivative(x):
        """Analytical derivative: f'(x) = 3x^2*sin(x) + x^3*cos(x)"""
        return 3 * x**2 * np.sin(x) + x**3 * np.cos(x)
    
    def finite_difference_naive(f, x, h):
        """Standard finite difference approximation."""
        return (f(x + h) - f(x - h)) / (2 * h)
    
    def finite_difference_kahan(f, x, h):
        """Finite difference with Kahan summation."""
        numerator_terms = [f(x + h), -f(x - h)]
        numerator = kahan_sum(numerator_terms)
        return numerator / (2 * h)
    
    # Test point and step sizes
    x0 = 2.0
    step_sizes = np.logspace(-16, -1, 16)
    
    print(f"Computing derivative of x^3 * sin(x) at x = {x0}")
    print(f"True derivative value: {true_derivative(x0):.10f}")
    print()
    
    print(f"{'Step Size':<12} {'Naive Error':<15} {'Kahan Error':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for h in step_sizes:
        # Calculate derivatives
        naive_deriv = finite_difference_naive(test_function, x0, h)
        kahan_deriv = finite_difference_kahan(test_function, x0, h)
        
        # Calculate errors
        true_deriv = true_derivative(x0)
        naive_error = abs(naive_deriv - true_deriv)
        kahan_error = abs(kahan_deriv - true_deriv)
        
        # Improvement factor
        if kahan_error > 0:
            improvement = naive_error / kahan_error
        else:
            improvement = float('inf')
        
        print(f"{h:<12.2e} {naive_error:<15.2e} {kahan_error:<15.2e} {improvement:<15.2f}")
    
    print()


def climate_simulation_example():
    """
    Climate model simulation with temperature averaging.
    
    Demonstrates precision issues in long-term climate data processing.
    """
    print("=" * 70)
    print("SCIENTIFIC EXAMPLE: Climate Data Processing")
    print("=" * 70)
    
    # Simulate 100 years of daily temperature data
    n_years = 100
    days_per_year = 365
    n_days = n_years * days_per_year
    
    print(f"Simulating {n_years} years of daily temperature data")
    print(f"Total data points: {n_days:,}")
    print()
    
    # Generate realistic temperature data
    np.random.seed(789)
    
    # Base temperature with seasonal variation
    day_of_year = np.arange(n_days) % days_per_year
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / days_per_year)
    
    # Add random daily variation
    daily_variation = np.random.normal(0, 5, n_days)
    
    # Add long-term trend (global warming)
    year_number = np.arange(n_days) // days_per_year
    long_term_trend = 0.02 * year_number  # 2°C per century
    
    # Combine all components
    temperatures = (seasonal_temp + daily_variation + long_term_trend).astype(np.float32)
    
    print(f"Temperature range: [{temperatures.min():.1f}°C, {temperatures.max():.1f}°C]")
    print()
    
    # Calculate long-term statistics
    def calculate_climate_statistics(temps):
        """Calculate climate statistics with different precision methods."""
        
        # Overall mean
        naive_mean = np.mean(temps)
        kahan_mean = compensated_mean(temps)
        einstein_mean, curvature = einstein_kahan_sum(temps)
        einstein_mean /= len(temps)
        
        # Decadal averages
        decades = []
        for decade in range(10):  # 10 decades
            start_idx = decade * days_per_year * 10
            end_idx = (decade + 1) * days_per_year * 10
            decade_temps = temps[start_idx:end_idx]
            
            decade_naive = np.mean(decade_temps)
            decade_kahan = compensated_mean(decade_temps)
            decade_einstein, _ = einstein_kahan_sum(decade_temps)
            decade_einstein /= len(decade_temps)
            
            decades.append((decade_naive, decade_kahan, decade_einstein))
        
        return (naive_mean, kahan_mean, einstein_mean, curvature), decades
    
    # Calculate statistics
    overall_stats, decadal_stats = calculate_climate_statistics(temperatures)
    
    print("Overall Temperature Statistics:")
    print("-" * 40)
    print(f"Naive mean:     {overall_stats[0]:.6f}°C")
    print(f"Kahan mean:     {overall_stats[1]:.6f}°C")
    print(f"Einstein mean:  {overall_stats[2]:.6f}°C")
    print(f"Curvature:      {overall_stats[3]:.2e}")
    print()
    
    print("Decadal Temperature Trends:")
    print("-" * 50)
    print(f"{'Decade':<8} {'Naive':<12} {'Kahan':<12} {'Einstein':<12}")
    print("-" * 50)
    
    for i, (naive, kahan, einstein) in enumerate(decadal_stats):
        decade_start = 1920 + i * 10
        print(f"{decade_start}s   {naive:<12.6f} {kahan:<12.6f} {einstein:<12.6f}")
    
    print()
    
    # Calculate long-term trend using linear regression on decadal means
    def calculate_trend(decade_means):
        """Calculate temperature trend in °C per decade."""
        x = np.arange(len(decade_means))
        
        # Use Kahan summation for regression calculations
        n = len(x)
        sum_x = kahan_sum(x)
        sum_y = kahan_sum(decade_means)
        sum_xy = kahan_sum(x * np.array(decade_means))
        sum_x2 = kahan_sum(x * x)
        
        # Calculate slope (trend)
        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x2 - sum_x * sum_x
        slope = numerator / denominator
        
        return slope
    
    trends = []
    for method_idx in range(3):
        decade_means = [decade[method_idx] for decade in decadal_stats]
        trend = calculate_trend(decade_means)
        trends.append(trend)
    
    print("Temperature Trends:")
    print("-" * 25)
    print(f"Naive trend:     {trends[0]:.4f}°C/decade")
    print(f"Kahan trend:     {trends[1]:.4f}°C/decade")
    print(f"Einstein trend:  {trends[2]:.4f}°C/decade")
    print(f"Expected trend:  0.2000°C/decade")
    print()


def main():
    """Run all scientific computing examples."""
    print("KAHAN SUMMATION LIBRARY - SCIENTIFIC COMPUTING EXAMPLES")
    print("=" * 70)
    print()
    
    particle_simulation_example()
    monte_carlo_integration()
    signal_processing_example()
    numerical_differentiation()
    climate_simulation_example()
    
    print("=" * 70)
    print("All scientific computing examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()