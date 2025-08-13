cmake_minimum_required(VERSION 3.14)
project(kahan_summation LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /arch:AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Wall")
endif()

# Find dependencies
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Include directories
include_directories(cpp)

# Source files
set(KAHAN_SOURCES
    cpp/kahan_kernel.cpp
)

set(KAHAN_HEADERS
    cpp/kahan_kernel.h
)

# Static library
add_library(kahan_static STATIC ${KAHAN_SOURCES})
target_include_directories(kahan_static PUBLIC cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kahan_static OpenMP::OpenMP_CXX)
endif()

# Shared library
add_library(kahan_shared SHARED ${KAHAN_SOURCES})
target_include_directories(kahan_shared PUBLIC cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kahan_shared OpenMP::OpenMP_CXX)
endif()

# Set library properties
set_target_properties(kahan_static PROPERTIES
    OUTPUT_NAME kahan
    CLEAN_DIRECT_OUTPUT 1
)

set_target_properties(kahan_shared PROPERTIES
    OUTPUT_NAME kahan
    CLEAN_DIRECT_OUTPUT 1
    VERSION 1.0.0
    SOVERSION 1
)

# Example executable
add_executable(kahan_example examples/cpp_example.cpp)
target_link_libraries(kahan_example kahan_static)

# Test executable
add_executable(kahan_test tests/cpp_tests.cpp)
target_link_libraries(kahan_test kahan_static)

# Install targets
install(TARGETS kahan_static kahan_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${KAHAN_HEADERS}
    DESTINATION include/kahan
)

# Python bindings (optional, requires pybind11)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    
    pybind11_add_module(kahan_cpp 
        cpp/python_bindings.cpp
        ${KAHAN_SOURCES}
    )
    
    target_compile_definitions(kahan_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
    
    if(OpenMP_CXX_FOUND)
        target_link_libraries(kahan_cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Testing with CTest
enable_testing()
add_test(NAME kahan_unit_tests COMMAND kahan_test)

# CPack configuration for packaging
set(CPACK_PACKAGE_NAME "kahan-summation")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-precision numerical summation library")
set(CPACK_PACKAGE_VENDOR "Kahan Summation Contributors")
set(CPACK_PACKAGE_CONTACT "contributors@kahan-summation.org")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Platform-specific packaging
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ;DragNDrop")
else()
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
endif()

include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "Kahan Summation Library Configuration Summary")
message(STATUS "===========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")