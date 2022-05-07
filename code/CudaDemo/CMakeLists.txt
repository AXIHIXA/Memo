cmake_minimum_required(VERSION 3.22)
project(CudaDemo CUDA)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(UTIL
        include/util/helper_cuda_drvapi.h
        include/util/helper_string.h
        )

set(MAIN
        src/main.cu
        )

set(ALL_SOURCE_FILES
        ${UTIL}
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
       ${CUDAToolkit_INCLUDE_DIRS}
        include
        )

# NOT all are needed!
set(CUDA_LIBRARIES
        # base
        CUDA::toolkit
        # CUDA Runtime
        CUDA::cudart
        CUDA::cudart_static
        # CUDA Driver
        CUDA::cuda_driver
        # cuBLAS
        CUDA::cublas
        CUDA::cublas_static
        CUDA::cublasLt
        CUDA::cublasLt_static
        # cuFFT
        CUDA::cufft
        CUDA::cufftw
        CUDA::cufft_static
#        CUDA::cufftw_static
        # cuRAND
        CUDA::curand
        CUDA::curand_static
        # cuSOLVER
        CUDA::cusolver
        CUDA::cusolver_static
        # cuSPARSE
        CUDA::cusparse
        CUDA::cusparse_static
        # cupti
        CUDA::cupti
        CUDA::cupti_static
        )

set(ALL_LIBRARIES
        ${CUDA_LIBRARIES}
        )

set(ALL_COMPILE_DEFS

        )

set(ALL_COMPILE_OPTS

        )

set(EXECUTABLE CudaDemo)
add_executable(${EXECUTABLE} ${ALL_SOURCE_FILES})
set_target_properties(${EXECUTABLE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${EXECUTABLE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${EXECUTABLE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${EXECUTABLE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})
