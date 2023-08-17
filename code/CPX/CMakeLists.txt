cmake_minimum_required(VERSION 3.27)

# CUDA 12 requires gcc<12 for compatibility issues.
# https://stackoverflow.com/questions/71448813/how-can-i-override-the-c-compiler-cmake-uses-for-cuda
# https://gist.github.com/ax3l/9489132
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-11")
project(CPX CUDA CXX)

# Ensure correct CUDA architecture
# NVIDIA Geforce RTX 2080 Ti has Compute Capability 7.5
# https://developer.nvidia.com/cuda-gpus
# https://stackoverflow.com/questions/67794606/cmake-cuda-architecture-flags
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

# If the main consumer of libfmt is a shared library (i.e., a Python extension),
# we must build libfmt as either shared library or position-independent static library.
# See https://fmt.dev/latest/usage.html for details.
find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt-10.0.0/build/")

set(Python_ROOT_DIR
        "$ENV{HOME}/opt/anaconda3/envs/py3/"
        )
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG HINTS "${Python_ROOT_DIR}/lib/python3.10/site-packages/pybind11/share/cmake/")

set(UTIL
        include/util/add.h
        include/util/fun.h
        include/util/pb.h
        src/util/add.cu
        src/util/fun.cpp
        src/util/pb.cpp
        )

set(MAIN
        src/main.cpp
        )

set(ALL_SOURCE_FILES
        ${MAIN}
        ${UTIL}
        )

set(ALL_INCLUDE_DIRS
        ${fmt_INCLUDE_DIRS}
        include
        )

set(ALL_LIBRARIES
        fmt::fmt
        pybind11::module
        pybind11::embed
        )

set(PYMODULE cpx)

set(ALL_COMPILE_DEFS
        -DMODULE_NAME=${PYMODULE}
        )

set(ALL_COMPILE_OPTS

        )

pybind11_add_module(${PYMODULE} ${ALL_SOURCE_FILES})
#add_executable(${PYMODULE} ${ALL_SOURCE_FILES})
target_compile_definitions(${PYMODULE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${PYMODULE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${PYMODULE} PUBLIC ${ALL_INCLUDE_DIRS})
#target_link_directories(${PYMODULE} PUBLIC "/usr/local/cuda/lib64/")
target_link_libraries(${PYMODULE} PUBLIC ${ALL_LIBRARIES})

# Note: Do NOT turn CUDA_SEPARABLE_COMPILATION on,
# otherwise there will be undefined symbols in the Python module!
#set_target_properties(${PYMODULE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
