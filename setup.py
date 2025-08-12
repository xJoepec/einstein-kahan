#!/usr/bin/env python3
"""
Setup script for Kahan Summation Library

This script builds both the Python package and optional C++ extensions
for high-performance numerical summation with error correction.
"""

import os
import sys
import subprocess
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

try:
    from setuptools import setup, Extension, find_packages
except ImportError:
    from distutils.core import setup, Extension
    find_packages = None

# Package metadata
PACKAGE_NAME = "kahan-summation"
VERSION = "1.0.0"
DESCRIPTION = "High-precision numerical summation with Kahan and Einstein-Kahan algorithms"
AUTHOR = "Kahan Summation Contributors"
AUTHOR_EMAIL = "contributors@kahan-summation.org"
URL = "https://github.com/your-username/kahan-summation"
LICENSE = "MIT"

# Read long description from README
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return DESCRIPTION

# Determine if we should build C++ extensions
def should_build_cpp():
    """Check if C++ extensions should be built."""
    if os.environ.get("KAHAN_NO_CPP", "").lower() in ("1", "true", "yes"):
        return False
    
    # Check for required compilers and tools
    try:
        subprocess.run(["c++", "--version"], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: C++ compiler not found, building without C++ extensions")
        return False

# C++ extension configuration
def get_cpp_extensions():
    """Get C++ extension configuration if building with C++."""
    if not should_build_cpp():
        return []
    
    # Source files
    cpp_sources = [
        "cpp/kahan_kernel.cpp",
        "cpp/python_bindings.cpp"
    ]
    
    # Compiler flags
    cpp_args = ["-O3", "-std=c++17"]
    link_args = []
    
    # Platform-specific optimizations
    if sys.platform.startswith("linux"):
        cpp_args.extend(["-march=native", "-fopenmp"])
        link_args.extend(["-fopenmp"])
    elif sys.platform == "darwin":  # macOS
        cpp_args.extend(["-march=native"])
        # Note: OpenMP support on macOS requires special handling
    elif sys.platform == "win32":
        cpp_args.extend(["/arch:AVX2", "/openmp"])
        link_args.extend(["/openmp"])
    
    # SIMD feature detection
    try:
        # Check for AVX2 support
        result = subprocess.run(
            ["python", "-c", "import cpuinfo; print('avx2' in cpuinfo.get_cpu_info()['flags'])"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip() == "True":
            cpp_args.append("-DAVX2_AVAILABLE")
    except:
        pass  # Fall back to runtime detection
    
    # Create extension
    ext_modules = [
        Pybind11Extension(
            "kahan.kahan_cpp",
            sources=cpp_sources,
            include_dirs=[
                pybind11.get_include(),
                "cpp/",
            ],
            language="c++",
            cxx_std=17,
            extra_compile_args=cpp_args,
            extra_link_args=link_args,
        ),
    ]
    
    return ext_modules

# Package requirements
def get_requirements():
    """Get package requirements."""
    base_requirements = [
        "numpy>=1.19.0",
        "torch>=1.9.0",
    ]
    
    dev_requirements = [
        "pytest>=6.0",
        "pytest-cov>=2.0",
        "black>=21.0",
        "flake8>=3.8",
        "mypy>=0.900",
        "sphinx>=4.0",
        "sphinx-rtd-theme>=1.0",
        "jupyter>=1.0",
        "matplotlib>=3.3",
        "scipy>=1.7",
    ]
    
    cpp_requirements = [
        "pybind11>=2.6.0",
    ]
    
    # Add C++ requirements if building extensions
    if should_build_cpp():
        base_requirements.extend(cpp_requirements)
    
    return {
        "base": base_requirements,
        "dev": dev_requirements,
        "cpp": cpp_requirements,
    }

# Custom build command that handles C++ compilation gracefully
class CustomBuildExt(build_ext):
    """Custom build extension that gracefully handles compilation failures."""
    
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
            print(f"Successfully built C++ extension: {ext.name}")
        except Exception as e:
            print(f"Warning: Failed to build C++ extension {ext.name}: {e}")
            print("Continuing with Python-only installation...")
            # Don't fail the entire build if C++ extensions fail

# Setup configuration
def main():
    """Main setup function."""
    requirements = get_requirements()
    
    # Package discovery
    if find_packages:
        packages = find_packages()
    else:
        packages = ["kahan"]
    
    # C++ extensions
    ext_modules = get_cpp_extensions()
    cmdclass = {}
    if ext_modules:
        cmdclass["build_ext"] = CustomBuildExt
    
    # Extras require for optional dependencies
    extras_require = {
        "dev": requirements["dev"],
        "cpp": requirements["cpp"],
        "all": requirements["dev"] + requirements["cpp"],
    }
    
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        
        # Package configuration
        packages=packages,
        package_dir={"kahan": "kahan"},
        
        # Dependencies
        install_requires=requirements["base"],
        extras_require=extras_require,
        python_requires=">=3.8",
        
        # C++ extensions
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        zip_safe=False,  # Required for C++ extensions
        
        # Metadata for PyPI
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9", 
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Operating System :: OS Independent",
        ],
        keywords=[
            "numerical", "summation", "kahan", "floating-point", 
            "precision", "error-correction", "parallel", "scientific-computing"
        ],
        
        # Project URLs
        project_urls={
            "Documentation": f"{URL}/docs",
            "Source": URL,
            "Tracker": f"{URL}/issues",
            "Changelog": f"{URL}/blob/main/CHANGELOG.md",
        },
        
        # Include additional files
        include_package_data=True,
        package_data={
            "kahan": ["*.pyi"],  # Type hints
        },
    )

if __name__ == "__main__":
    main()