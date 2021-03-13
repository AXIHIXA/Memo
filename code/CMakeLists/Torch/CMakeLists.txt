cmake_minimum_required(VERSION 3.17)
project(TorchCpp)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# CMAKE_PREFIX_PATH can be gotten from `python -m "import torch;print(torch.utils.cmake_prefix_path)"`
set(CMAKE_PREFIX_PATH
        /home/xihan1/opt/anaconda3/envs/py3/lib/python3.7/site-packages/torch/share/cmake
        ${CMAKE_PREFIX_PATH}
        )

find_package(Torch REQUIRED)

set(MAIN
        src/main.cpp
        )

set(ALL_SOURCE_FILES
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        include
        )

set(ALL_LIBRARIES
        ${TORCH_LIBRARIES}
        )

set(ALL_COMPILE_DEFS

        )

set(ALL_COMPILE_OPTS
        -Wpessimizing-move
        -Wredundant-move
        )

add_executable(${PROJECT_NAME} ${ALL_SOURCE_FILES})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${PROJECT_NAME} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBRARIES})
