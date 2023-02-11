cmake_minimum_required(VERSION 3.23)
project(CudaDemo CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

find_package(fmt REQUIRED HINTS "/home/xihan1/lib/fmt-9.1.0/build/")

set(UTIL
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
        fmt::fmt
        )

set(ALL_COMPILE_DEFS

        )

set(ALL_COMPILE_OPTS

        )

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${ALL_SOURCE_FILES})
set_target_properties(${EXECUTABLE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${EXECUTABLE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${EXECUTABLE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${EXECUTABLE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})

