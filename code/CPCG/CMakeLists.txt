cmake_minimum_required(VERSION 3.27)

project(CPCG CUDA CXX)

# Ensure correct CUDA architecture
# NVIDIA Geforce RTX 2080 Ti has Compute Capability 7.5
# https://developer.nvidia.com/cuda-gpus
# https://stackoverflow.com/questions/67794606/cmake-cuda-architecture-flags
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Boost REQUIRED)

set(Python_ROOT_DIR "$ENV{HOME}/opt/anaconda3/envs/py3/")
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG HINTS "${Python_ROOT_DIR}/lib/python3.10/site-packages/pybind11/share/cmake/")

set(TBB_LIBRARIES
        # Do NOT include libtbbmalloc and libtbbmaloc_proxy, otherwise crashes after exiting Python intepreter
        /usr/lib/x86_64-linux-gnu/libtbb.so
)

set(CPCG
        include/cpcg/cpcg.h
        src/cpcg/cpcg.cu
)

set(MAIN
        src/main.cpp
)

set(ALL_SOURCE_FILES
        ${CPCG}
        ${MAIN}
)

set(ALL_INCLUDE_DIRS
        ${fmt_INCLUDE_DIRS}
        include
)

set(ALL_LIBRARIES
        ${TBB_LIBRARIES}
        Boost::boost
        cudart
        cublas
        cusparse
        pybind11::module
)

set(PYMODULE cpcg)

set(ALL_COMPILE_DEFS
        -DMODULE_NAME=${PYMODULE}
)

set(ALL_COMPILE_OPTS

)

pybind11_add_module(${PYMODULE} ${ALL_SOURCE_FILES})
target_compile_definitions(${PYMODULE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${PYMODULE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${PYMODULE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PYMODULE} PUBLIC ${ALL_LIBRARIES})

# NOT NEEDED given CMAKE_CUDA_ARCHITECTURE is set properly
# target_compile_options(${EXECUTABLE} PRIVATE
#         $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_75,code=[compute_75,sm_75]>)
