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

find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt/build")

set(UTIL
        include/util/CudaUtil.h
        include/util/TimerGuard.h
)

set(MAIN
        src/main.cu
)

set(ALL_SOURCE_FILES
        ${UTIL}
        ${MAIN}
)

set(ALL_INCLUDE_DIRS
        include
)

set(ALL_LIBRARIES
        cudart
        cublas
        curand
        cusparse
        fmt::fmt
)

set(ALL_COMPILE_DEFS

)

set(ALL_COMPILE_OPTS
        # -g is default in CMAKE_<LANG>_FLAGS_DEBUG
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-O0>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

set(EXECUTABLE exe)
add_executable(${EXECUTABLE} ${ALL_SOURCE_FILES})
target_compile_definitions(${EXECUTABLE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${EXECUTABLE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${EXECUTABLE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})

# NOT NEEDED given CMAKE_CUDA_ARCHITECTURE is set properly
# target_compile_options(${EXECUTABLE} PRIVATE
#         $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_75,code=[compute_75,sm_75]>)


