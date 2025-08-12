#include "kahan_kernel.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace kahan {

// SimdKahanSum implementation

#ifdef __AVX2__
double SimdKahanSum::sum_avx2(const float* data, size_t n, double* error_out) {
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d err_vec = _mm256_setzero_pd();
    
    size_t i = 0;
    // Process 8 floats at a time (2 AVX2 256-bit vectors of 4 doubles each)
    for (; i + 8 <= n; i += 8) {
        // Load 8 floats and convert to 2 vectors of 4 doubles
        __m128 float_vec1 = _mm_loadu_ps(data + i);
        __m128 float_vec2 = _mm_loadu_ps(data + i + 4);
        
        __m256d val_vec1 = _mm256_cvtps_pd(float_vec1);
        __m256d val_vec2 = _mm256_cvtps_pd(float_vec2);
        
        // Kahan summation for first 4 elements
        __m256d y1 = _mm256_sub_pd(val_vec1, err_vec);
        __m256d t1 = _mm256_add_pd(sum_vec, y1);
        err_vec = _mm256_sub_pd(_mm256_sub_pd(t1, sum_vec), y1);
        sum_vec = t1;
        
        // Kahan summation for next 4 elements
        __m256d y2 = _mm256_sub_pd(val_vec2, err_vec);
        __m256d t2 = _mm256_add_pd(sum_vec, y2);
        err_vec = _mm256_sub_pd(_mm256_sub_pd(t2, sum_vec), y2);
        sum_vec = t2;
    }
    
    // Horizontal sum of the vector
    double sum_arr[4], err_arr[4];
    _mm256_storeu_pd(sum_arr, sum_vec);
    _mm256_storeu_pd(err_arr, err_vec);
    
    KahanAccumulator final_acc;
    for (int j = 0; j < 4; ++j) {
        final_acc.add(sum_arr[j]);
    }
    
    // Add accumulated error
    for (int j = 0; j < 4; ++j) {
        final_acc.error += err_arr[j];
    }
    
    // Handle remaining elements
    for (; i < n; ++i) {
        final_acc.add(static_cast<double>(data[i]));
    }
    
    if (error_out) *error_out = final_acc.error;
    return final_acc.get();
}
#endif

double SimdKahanSum::sum_scalar(const float* data, size_t n, double* error_out) {
    KahanAccumulator acc;
    
    for (size_t i = 0; i < n; ++i) {
        acc.add(static_cast<double>(data[i]));
    }
    
    if (error_out) *error_out = acc.error;
    return acc.get();
}

double SimdKahanSum::sum_auto(const float* data, size_t n, double* error_out) {
    #ifdef __AVX2__
    return sum_avx2(data, n, error_out);
    #else
    return sum_scalar(data, n, error_out);
    #endif
}

// TreeKahanReduction implementation

std::pair<double, double> TreeKahanReduction::reduce_range(const float* data, size_t start, size_t end) {
    if (start >= end) return {0.0, 0.0};
    if (end - start == 1) return {static_cast<double>(data[start]), 0.0};
    
    // Base case for small ranges
    if (end - start <= 64) {
        double error;
        double sum = SimdKahanSum::sum_scalar(data + start, end - start, &error);
        return {sum, error};
    }
    
    // Recursive tree reduction
    size_t mid = start + (end - start) / 2;
    auto left_result = reduce_range(data, start, mid);
    auto right_result = reduce_range(data, mid, end);
    
    // Combine results with Kahan compensation
    KahanAccumulator acc;
    acc.add(left_result.first);
    acc.add(right_result.first);
    
    double total_error = left_result.second + right_result.second + acc.error;
    return {acc.get(), total_error};
}

std::pair<double, double> TreeKahanReduction::reduce(const float* data, size_t n, int num_threads) {
    if (n == 0) return {0.0, 0.0};
    
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    
    if (num_threads == 1 || n < 1000) {
        // Single-threaded for small arrays
        return reduce_range(data, 0, n);
    }
    
    // Parallel reduction
    std::vector<std::pair<double, double>> partial_results(num_threads);
    
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        size_t chunk_size = n / num_threads;
        size_t start = thread_id * chunk_size;
        size_t end = (thread_id == num_threads - 1) ? n : start + chunk_size;
        
        partial_results[thread_id] = reduce_range(data, start, end);
    }
    
    // Combine partial results
    KahanAccumulator final_acc;
    double total_error = 0.0;
    
    for (const auto& result : partial_results) {
        final_acc.add(result.first);
        total_error += result.second;
    }
    
    total_error += final_acc.error;
    return {final_acc.get(), total_error};
}

std::pair<double, double> TreeKahanReduction::reduce_blocked(const float* data, size_t n, size_t block_size) {
    if (n == 0) return {0.0, 0.0};
    if (block_size == 0) block_size = 1024; // Default block size
    
    std::vector<std::pair<double, double>> block_results;
    
    for (size_t i = 0; i < n; i += block_size) {
        size_t end = std::min(i + block_size, n);
        block_results.push_back(reduce_range(data, i, end));
    }
    
    // Tree reduce the block results
    while (block_results.size() > 1) {
        std::vector<std::pair<double, double>> next_level;
        
        for (size_t i = 0; i < block_results.size(); i += 2) {
            if (i + 1 < block_results.size()) {
                KahanAccumulator acc;
                acc.add(block_results[i].first);
                acc.add(block_results[i + 1].first);
                
                double combined_error = block_results[i].second + 
                                       block_results[i + 1].second + 
                                       acc.error;
                next_level.push_back({acc.get(), combined_error});
            } else {
                next_level.push_back(block_results[i]);
            }
        }
        
        block_results = std::move(next_level);
    }
    
    return block_results.empty() ? std::make_pair(0.0, 0.0) : block_results[0];
}

// EinsteinKahanSummer implementation

std::pair<double, double> EinsteinKahanSummer::sum(const float* data, size_t n, double einstein_constant) {
    if (n == 0) return {0.0, 0.0};
    
    // Compute standard Kahan sum
    double error;
    double kahan_sum = SimdKahanSum::sum_auto(data, n, &error);
    
    // Compute curvature error
    double curvature = compute_curvature(data, n);
    
    // Apply Einstein geometric correction
    double correction = apply_correction(kahan_sum, curvature, n, einstein_constant);
    
    return {correction, curvature};
}

double EinsteinKahanSummer::compute_curvature(const float* data, size_t n) {
    if (n == 0) return 0.0;
    
    // Compute true sum in higher precision
    double true_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        true_sum += static_cast<double>(data[i]);
    }
    
    // Compute Kahan sum
    double kahan_result = SimdKahanSum::sum_auto(data, n);
    
    // Curvature is the difference
    return true_sum - kahan_result;
}

double EinsteinKahanSummer::apply_correction(double raw_sum, double curvature, size_t n, double einstein_constant) {
    if (n == 0) return raw_sum;
    
    // Geometric correction inspired by Einstein's field equations
    double correction = einstein_constant * curvature / static_cast<double>(n);
    
    return raw_sum + correction;
}

// BatchKahanOperations implementation

void BatchKahanOperations::batch_sum(const float** arrays, const size_t* sizes, size_t count, 
                                    double* results, double* errors) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        double error;
        results[i] = SimdKahanSum::sum_auto(arrays[i], sizes[i], &error);
        if (errors) errors[i] = error;
    }
}

double BatchKahanOperations::pairwise_sum(const float* a, const float* b, size_t n) {
    KahanAccumulator acc;
    
    for (size_t i = 0; i < n; ++i) {
        acc.add(static_cast<double>(a[i]) + static_cast<double>(b[i]));
    }
    
    return acc.get();
}

// Utility functions

namespace utils {

double estimate_condition_number(const float* data, size_t n) {
    if (n == 0) return 1.0;
    
    double max_val = 0.0;
    double sum_abs = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double abs_val = std::abs(static_cast<double>(data[i]));
        max_val = std::max(max_val, abs_val);
        sum_abs += abs_val;
    }
    
    if (max_val == 0.0) return 1.0;
    return sum_abs / max_val;
}

int recommend_algorithm(const float* data, size_t n, double target_precision) {
    if (n == 0) return 0;
    
    double condition_number = estimate_condition_number(data, n);
    double estimated_error = condition_number * std::numeric_limits<float>::epsilon();
    
    if (estimated_error < target_precision) {
        return 0; // Naive summation sufficient
    } else if (estimated_error < target_precision * 10) {
        return 1; // Kahan summation
    } else if (estimated_error < target_precision * 100) {
        return 2; // Einstein-Kahan
    } else {
        return 3; // Tree reduction
    }
}

} // namespace utils

// C interface implementation

extern "C" {

double kahan_sum_c(const float* data, size_t n) {
    return SimdKahanSum::sum_auto(data, n);
}

double einstein_kahan_sum_c(const float* data, size_t n, double* error_out) {
    auto result = EinsteinKahanSummer::sum(data, n);
    if (error_out) *error_out = result.second;
    return result.first;
}

double tree_reduce_c(const float* data, size_t n, double* error_out) {
    auto result = TreeKahanReduction::reduce(data, n);
    if (error_out) *error_out = result.second;
    return result.first;
}

void batch_sum_c(const float** arrays, const size_t* sizes, size_t count, double* results) {
    BatchKahanOperations::batch_sum(arrays, sizes, count, results);
}

} // extern "C"

} // namespace kahan