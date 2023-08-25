cmake_minimum_required(VERSION 3.27)

# CUDA 12 requires gcc<12 for compatibility issues.
# https://stackoverflow.com/questions/71448813/how-can-i-override-the-c-compiler-cmake-uses-for-cuda
# https://gist.github.com/ax3l/9489132
# set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-11")
project(PTE CUDA CXX)

# Ensure correct CUDA architecture
# NVIDIA Geforce RTX 2080 Ti has Compute Capability 7.5
# https://developer.nvidia.com/cuda-gpus
# https://stackoverflow.com/questions/67794606/cmake-cuda-architecture-flags
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(VERBOSE ON)

# If the main consumer of libfmt is a shared library (i.e., a Python extension),
# we must build libfmt as either shared library or position-independent static library.
# See https://fmt.dev/latest/usage.html for details.
find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt-10.0.0/build/")

set(Python_ROOT_DIR "$ENV{HOME}/opt/anaconda3/envs/py3/")
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG HINTS "${Python_ROOT_DIR}/lib/python3.10/site-packages/pybind11/share/cmake/")

# CMAKE_PREFIX_PATH can be gotten from `python -m "import torch;print(torch.utils.cmake_prefix_path)"`
# libTorch conda build conflicts with OpenCV, so download compiled library directly from pytorch.org.
# libtorch REQUIRES CMAKE_CUDA_STANDARD <= 17 and CMAKE_CXX_STANDARD <= 17.
set(CAFFE2_USE_CUDNN ON)
find_package(Torch REQUIRED CONFIG HINTS "$ENV{HOME}/opt/anaconda3/envs/py3/lib/python3.10/site-packages/torch/share/cmake/")

# Conda TORCH_CXX_FLAGS contains -D_GLIBCXX_USE_CXX11_ABI=1,
# while the conda build is compiled with pre-cxx11 abi!
# (Test with command $ python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)".)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TBB_LIBRARIES
        # Do NOT include libtbbmalloc and libtbbmaloc_proxy, otherwise crashes after exiting Python intepreter
        /usr/lib/x86_64-linux-gnu/libtbb.so
        )

set(UTIL
        include/util/pybind11_global.h
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
        ${TORCH_INCLUDE_DIRS}
        include
        )

set(ALL_LIBRARIES
        ${TBB_LIBRARIES}
        ${TORCH_LIBRARIES}
        pybind11::module
        fmt::fmt
        )

set(TORCH_EXTENSION pte)

set(ALL_COMPILE_DEFS
        -DTORCH_EXTENSION_NAME=${TORCH_EXTENSION}
        )

set(ALL_COMPILE_OPTS
        # Conda pytorch build uses pre-cxx11 abi
        -D_GLIBCXX_USE_CXX11_ABI=0
        )

pybind11_add_module(${TORCH_EXTENSION} SHARED ${ALL_SOURCE_FILES})
target_compile_definitions(${TORCH_EXTENSION} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${TORCH_EXTENSION} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${TORCH_EXTENSION} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${TORCH_EXTENSION} PUBLIC ${ALL_LIBRARIES})

# Note: Do NOT turn CUDA_SEPARABLE_COMPILATION on,
# otherwise there will be undefined symbols in the Python module!
#set_target_properties(${TORCH_EXTENSION} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
