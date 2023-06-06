cmake_minimum_required(VERSION 3.23)
project(CudaDemo CUDA CXX)
# Ensure correct CUDA architecture
# NVIDIA Geforce RTX 2080 Ti has Compute Capability 7.5
# https://developer.nvidia.com/cuda-gpus
# https://stackoverflow.com/questions/67794606/cmake-cuda-architecture-flags
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt-9.1.0/build")

set(CUAX
        include/cuax/pccg.h
        src/cuax/pccg.cu
        )

set(UTIL
        include/util/TimerGuard.h
        src/util/TimerGuard.cpp
        )

set(MAIN
        src/main.cpp
        )

set(ALL_SOURCE_FILES
        ${CUAX}
        ${UTIL}
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        include
        )

set(ALL_LIBRARIES
        Eigen3::Eigen
        fmt::fmt
        )

set(ALL_CXX_COMPILE_DEFS

        )

set(ALL_CXX_COMPILE_OPTS

        )

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${ALL_SOURCE_FILES})
set_target_properties(${EXECUTABLE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${EXECUTABLE} PUBLIC ${ALL_CXX_COMPILE_DEFS})
target_compile_options(${EXECUTABLE} PUBLIC ${ALL_CXX_COMPILE_OPTS})
target_include_directories(${EXECUTABLE} BEFORE PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})

# NOT NEEDED given CMAKE_CUDA_ARCHITECTURE is set properly
# target_compile_options(${EXECUTABLE} PRIVATE
#         $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_70,code=[compute_70,sm_70]>)
