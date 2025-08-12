#ifndef KAHAN_KERNEL_H
#define KAHAN_KERNEL_H

#include <cmath>
#include <vector>
#include <immintrin.h>  // For SIMD intrinsics
#include <omp.h>        // For OpenMP parallelization

namespace kahan {

// Constants from geometric theory
constexpr double EINSTEIN_8PI = 8.0 * M_PI;
constexpr double NUMERICAL_EPSILON = 1e-15;

/**
 * Structure to hold sum and error together for Kahan summation
 */
struct KahanAccumulator {
    double sum;
    double error;
    
    KahanAccumulator() : sum(0.0), error(0.0) {}
    
    /**
     * Add a value with Kahan error compensation
     * @param value Value to add
     */
    inline void add(double value) {
        double y = value - error;
        double t = sum + y;
        error = (t - sum) - y;
        sum = t;
    }
    
    /**
     * Get the compensated sum
     * @return Corrected sum value
     */
    inline double get() const {
        return sum;
    }
    
    /**
     * Get sum with Einstein geometric correction
     * @return Sum with 8π error correction applied
     */
    inline double get_corrected_sum() const {
        return sum + EINSTEIN_8PI * error;
    }
    
    /**
     * Reset accumulator to zero
     */
    inline void reset() {
        sum = 0.0;
        error = 0.0;
    }
};

/**
 * SIMD-optimized Kahan summation for arrays
 */
class SimdKahanSum {
public:
    // AVX-512 version for modern x86_64
    #ifdef __AVX512F__
    static double sum_avx512(const float* data, size_t n, double* error_out = nullptr);
    #endif
    
    // AVX2 version for standard x86_64
    #ifdef __AVX2__
    static double sum_avx2(const float* data, size_t n, double* error_out = nullptr);
    #endif
    
    // Fallback scalar version
    static double sum_scalar(const float* data, size_t n, double* error_out = nullptr);
    
    // Auto-dispatching version
    static double sum_auto(const float* data, size_t n, double* error_out = nullptr);
};

/**
 * Tree-based parallel reduction with Kahan compensation
 */
class TreeKahanReduction {
public:
    /**
     * Perform tree reduction on array
     * @param data Input array
     * @param n Array size
     * @param num_threads Number of OpenMP threads to use
     * @return Tuple of (sum, total_error)
     */
    static std::pair<double, double> reduce(const float* data, size_t n, int num_threads = 0);
    
    /**
     * Parallel reduction with custom block size
     * @param data Input array
     * @param n Array size
     * @param block_size Size of each processing block
     * @return Tuple of (sum, total_error)
     */
    static std::pair<double, double> reduce_blocked(const float* data, size_t n, size_t block_size);

private:
    static std::pair<double, double> reduce_range(const float* data, size_t start, size_t end);
};

/**
 * Einstein-Kahan summation with geometric error correction
 */
class EinsteinKahanSummer {
public:
    /**
     * Perform Einstein-Kahan summation
     * @param data Input array
     * @param n Array size
     * @param einstein_constant Geometric normalization constant (default: 8π)
     * @return Tuple of (corrected_sum, curvature_error)
     */
    static std::pair<double, double> sum(
        const float* data, 
        size_t n, 
        double einstein_constant = EINSTEIN_8PI
    );
    
    /**
     * Compute geometric curvature of the summation path
     * @param data Input array
     * @param n Array size
     * @return Estimated curvature error
     */
    static double compute_curvature(const float* data, size_t n);
    
    /**
     * Apply Einstein correction to a raw sum
     * @param raw_sum Uncorrected sum
     * @param curvature Computed curvature error
     * @param n Number of elements
     * @param einstein_constant Geometric constant
     * @return Corrected sum
     */
    static double apply_correction(
        double raw_sum, 
        double curvature, 
        size_t n, 
        double einstein_constant = EINSTEIN_8PI
    );
};

/**
 * Batch operations for multiple arrays
 */
class BatchKahanOperations {
public:
    /**
     * Sum multiple arrays in batch
     * @param arrays Array of pointers to input arrays
     * @param sizes Array of sizes for each input array
     * @param count Number of arrays
     * @param results Output array for results
     * @param errors Output array for error estimates (optional)
     */
    static void batch_sum(
        const float** arrays,
        const size_t* sizes,
        size_t count,
        double* results,
        double* errors = nullptr
    );
    
    /**
     * Compute pairwise sums efficiently
     * @param a First array
     * @param b Second array  
     * @param n Array size (must be same for both)
     * @return Sum of element-wise sums
     */
    static double pairwise_sum(const float* a, const float* b, size_t n);
};

/**
 * Utility functions for numerical stability
 */
namespace utils {
    /**
     * Check for numerical instability
     * @param value Value to check
     * @return True if value is unstable (NaN, inf, or extreme)
     */
    inline bool is_unstable(double value) {
        return std::isnan(value) || std::isinf(value) || 
               std::abs(value) > 1e15 || std::abs(value) < NUMERICAL_EPSILON;
    }
    
    /**
     * Stabilize value if needed
     * @param value Input value
     * @return Stabilized value
     */
    inline double stabilize(double value) {
        if (std::isnan(value)) return 0.0;
        if (std::isinf(value)) return value > 0 ? 1e15 : -1e15;
        if (std::abs(value) < NUMERICAL_EPSILON) return 0.0;
        return value;
    }
    
    /**
     * Estimate condition number for summation
     * @param data Input array
     * @param n Array size
     * @return Estimated condition number
     */
    double estimate_condition_number(const float* data, size_t n);
    
    /**
     * Determine optimal algorithm based on data characteristics
     * @param data Input array
     * @param n Array size
     * @param target_precision Desired precision
     * @return Recommended algorithm (0=naive, 1=kahan, 2=einstein, 3=tree)
     */
    int recommend_algorithm(const float* data, size_t n, double target_precision = 1e-15);
}

/**
 * C interface for Python bindings
 */
extern "C" {
    /**
     * C interface for Kahan summation
     */
    double kahan_sum_c(const float* data, size_t n);
    
    /**
     * C interface for Einstein-Kahan summation
     */
    double einstein_kahan_sum_c(const float* data, size_t n, double* error_out);
    
    /**
     * C interface for tree reduction
     */
    double tree_reduce_c(const float* data, size_t n, double* error_out);
    
    /**
     * C interface for batch operations
     */
    void batch_sum_c(const float** arrays, const size_t* sizes, size_t count, double* results);
}

} // namespace kahan

#endif // KAHAN_KERNEL_H