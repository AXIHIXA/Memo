cmake_minimum_required(VERSION 3.23)
project(MVCP CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Boost REQUIRED)
find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt/build/")

# CMAKE_PREFIX_PATH can be gotten from `python -m "import torch;print(torch.utils.cmake_prefix_path)"`
# libTorch conda build conflicts with OpenCV, so download compiled library directly from pytorch.org.
# libtorch REQUIRES CMAKE_CUDA_STANDARD <= 17 and CMAKE_CXX_STANDARD <= 17.
set(CAFFE2_USE_CUDNN ON)
find_package(Torch REQUIRED CONFIG HINTS "$ENV{HOME}/lib/libtorch/share/cmake/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TBB_LIBRARIES
        # Do NOT include libtbbmalloc and libtbbmaloc_proxy, otherwise crashes after exiting Python intepreter
        /usr/lib/x86_64-linux-gnu/libtbb.so
)

set(MODEL
        include/model/MVP.h
        src/model/MVP.cpp
)

set(UTIL
        include/util/Formatter.h
        include/util/TimerGuard.h
)

set(MAIN
        src/main.cpp
)

set(ALL_SOURCE_FILES
        ${MODEL}
        ${UTIL}
        ${MAIN}
)

set(ALL_INCLUDE_DIRS
        ${TORCH_INCLUDE_DIRS}
        include
)

set(ALL_LIBRARIES
        ${Boost_LIBRARIES}
        ${TBB_LIBRARIES}
        ${TORCH_LIBRARIES}
        fmt::fmt
        pthread
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
target_include_directories(${EXECUTABLE} BEFORE PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})
