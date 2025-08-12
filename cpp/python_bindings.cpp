#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "kahan_kernel.h"

namespace py = pybind11;
using namespace kahan;

/**
 * Convert numpy array to pointer and size
 */
template<typename T>
std::pair<const T*, size_t> get_array_info(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    return {static_cast<const T*>(buf.ptr), static_cast<size_t>(buf.size)};
}

/**
 * Python wrapper for Kahan summation
 */
double py_kahan_sum(py::array_t<float> input) {
    auto [data, n] = get_array_info(input);
    return SimdKahanSum::sum_auto(data, n);
}

/**
 * Python wrapper for Kahan summation with error tracking
 */
std::pair<double, double> py_kahan_sum_with_error(py::array_t<float> input) {
    auto [data, n] = get_array_info(input);
    double error;
    double sum = SimdKahanSum::sum_auto(data, n, &error);
    return {sum, error};
}

/**
 * Python wrapper for Einstein-Kahan summation
 */
std::pair<double, double> py_einstein_kahan_sum(py::array_t<float> input, double einstein_constant = EINSTEIN_8PI) {
    auto [data, n] = get_array_info(input);
    return EinsteinKahanSummer::sum(data, n, einstein_constant);
}

/**
 * Python wrapper for tree reduction
 */
std::pair<double, double> py_tree_reduce(py::array_t<float> input, int num_threads = 0) {
    auto [data, n] = get_array_info(input);
    return TreeKahanReduction::reduce(data, n, num_threads);
}

/**
 * Python wrapper for tree reduction with custom block size
 */
std::pair<double, double> py_tree_reduce_blocked(py::array_t<float> input, size_t block_size = 1024) {
    auto [data, n] = get_array_info(input);
    return TreeKahanReduction::reduce_blocked(data, n, block_size);
}

/**
 * Python wrapper for batch summation
 */
py::array_t<double> py_batch_sum(py::list input_arrays) {
    size_t count = input_arrays.size();
    if (count == 0) {
        return py::array_t<double>(0);
    }
    
    // Prepare data structures
    std::vector<const float*> arrays(count);
    std::vector<size_t> sizes(count);
    
    for (size_t i = 0; i < count; ++i) {
        py::array_t<float> arr = input_arrays[i].cast<py::array_t<float>>();
        auto [data, n] = get_array_info(arr);
        arrays[i] = data;
        sizes[i] = n;
    }
    
    // Allocate output
    auto result = py::array_t<double>(count);
    py::buffer_info result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Perform batch summation
    BatchKahanOperations::batch_sum(arrays.data(), sizes.data(), count, result_ptr);
    
    return result;
}

/**
 * Python wrapper for batch summation with error tracking
 */
std::pair<py::array_t<double>, py::array_t<double>> py_batch_sum_with_errors(py::list input_arrays) {
    size_t count = input_arrays.size();
    if (count == 0) {
        return {py::array_t<double>(0), py::array_t<double>(0)};
    }
    
    // Prepare data structures
    std::vector<const float*> arrays(count);
    std::vector<size_t> sizes(count);
    
    for (size_t i = 0; i < count; ++i) {
        py::array_t<float> arr = input_arrays[i].cast<py::array_t<float>>();
        auto [data, n] = get_array_info(arr);
        arrays[i] = data;
        sizes[i] = n;
    }
    
    // Allocate outputs
    auto results = py::array_t<double>(count);
    auto errors = py::array_t<double>(count);
    
    py::buffer_info results_buf = results.request();
    py::buffer_info errors_buf = errors.request();
    
    double* results_ptr = static_cast<double*>(results_buf.ptr);
    double* errors_ptr = static_cast<double*>(errors_buf.ptr);
    
    // Perform batch summation with error tracking
    BatchKahanOperations::batch_sum(arrays.data(), sizes.data(), count, results_ptr, errors_ptr);
    
    return {results, errors};
}

/**
 * Python wrapper for pairwise sum
 */
double py_pairwise_sum(py::array_t<float> a, py::array_t<float> b) {
    auto [data_a, n_a] = get_array_info(a);
    auto [data_b, n_b] = get_array_info(b);
    
    if (n_a != n_b) {
        throw std::runtime_error("Arrays must have the same size");
    }
    
    return BatchKahanOperations::pairwise_sum(data_a, data_b, n_a);
}

/**
 * Python wrapper for condition number estimation
 */
double py_estimate_condition_number(py::array_t<float> input) {
    auto [data, n] = get_array_info(input);
    return utils::estimate_condition_number(data, n);
}

/**
 * Python wrapper for algorithm recommendation
 */
int py_recommend_algorithm(py::array_t<float> input, double target_precision = 1e-15) {
    auto [data, n] = get_array_info(input);
    return utils::recommend_algorithm(data, n, target_precision);
}

/**
 * Python wrapper for curvature computation
 */
double py_compute_curvature(py::array_t<float> input) {
    auto [data, n] = get_array_info(input);
    return EinsteinKahanSummer::compute_curvature(data, n);
}

/**
 * KahanAccumulator Python wrapper class
 */
class PyKahanAccumulator {
private:
    KahanAccumulator acc;
    
public:
    PyKahanAccumulator() = default;
    
    void add(double value) {
        acc.add(value);
    }
    
    void add_array(py::array_t<float> input) {
        auto [data, n] = get_array_info(input);
        for (size_t i = 0; i < n; ++i) {
            acc.add(static_cast<double>(data[i]));
        }
    }
    
    double get() const {
        return acc.get();
    }
    
    double get_corrected() const {
        return acc.get_corrected_sum();
    }
    
    double get_error() const {
        return acc.error;
    }
    
    void reset() {
        acc.reset();
    }
};

PYBIND11_MODULE(kahan_cpp, m) {
    m.doc() = "High-performance Kahan summation library with C++ backend";
    
    // Constants
    m.attr("EINSTEIN_8PI") = EINSTEIN_8PI;
    m.attr("NUMERICAL_EPSILON") = NUMERICAL_EPSILON;
    
    // Basic summation functions
    m.def("kahan_sum", &py_kahan_sum, 
          "Compute sum using Kahan compensated summation",
          py::arg("input"));
    
    m.def("kahan_sum_with_error", &py_kahan_sum_with_error,
          "Compute sum using Kahan summation with error tracking",
          py::arg("input"));
    
    m.def("einstein_kahan_sum", &py_einstein_kahan_sum,
          "Compute sum using Einstein-Kahan geometric error correction",
          py::arg("input"), py::arg("einstein_constant") = EINSTEIN_8PI);
    
    m.def("tree_reduce", &py_tree_reduce,
          "Perform tree-based parallel reduction with Kahan compensation",
          py::arg("input"), py::arg("num_threads") = 0);
    
    m.def("tree_reduce_blocked", &py_tree_reduce_blocked,
          "Perform tree reduction with custom block size",
          py::arg("input"), py::arg("block_size") = 1024);
    
    // Batch operations
    m.def("batch_sum", &py_batch_sum,
          "Sum multiple arrays in batch",
          py::arg("input_arrays"));
    
    m.def("batch_sum_with_errors", &py_batch_sum_with_errors,
          "Sum multiple arrays in batch with error tracking",
          py::arg("input_arrays"));
    
    m.def("pairwise_sum", &py_pairwise_sum,
          "Compute sum of element-wise array additions",
          py::arg("a"), py::arg("b"));
    
    // Utility functions
    m.def("estimate_condition_number", &py_estimate_condition_number,
          "Estimate condition number for summation accuracy",
          py::arg("input"));
    
    m.def("recommend_algorithm", &py_recommend_algorithm,
          "Recommend optimal summation algorithm based on data characteristics",
          py::arg("input"), py::arg("target_precision") = 1e-15);
    
    m.def("compute_curvature", &py_compute_curvature,
          "Compute geometric curvature error for the summation",
          py::arg("input"));
    
    // KahanAccumulator class
    py::class_<PyKahanAccumulator>(m, "KahanAccumulator")
        .def(py::init<>())
        .def("add", &PyKahanAccumulator::add, "Add a single value")
        .def("add_array", &PyKahanAccumulator::add_array, "Add all values from an array")
        .def("get", &PyKahanAccumulator::get, "Get the current sum")
        .def("get_corrected", &PyKahanAccumulator::get_corrected, "Get sum with Einstein correction")
        .def("get_error", &PyKahanAccumulator::get_error, "Get accumulated error")
        .def("reset", &PyKahanAccumulator::reset, "Reset accumulator to zero");
    
    // Algorithm recommendation constants
    py::enum_<int>(m, "Algorithm")
        .value("NAIVE", 0)
        .value("KAHAN", 1)
        .value("EINSTEIN_KAHAN", 2)
        .value("TREE_REDUCE", 3);
}