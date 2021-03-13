cmake_minimum_required(VERSION 3.16)
project(SketchBoard)
set(CMAKE_CXX_STANDARD 14)

# These 3 options for Qt support
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# CMAKE_PREFIX_PATH can be gotten from `python -m "import torch;print(torch.utils.cmake_prefix_path)"`
# libTorch conda build conflicts with OpenCV, so download compiled library directly from pytorch.org.
set(CMAKE_PREFIX_PATH
        /home/xihan1/lib/libtorch/share/cmake
        ${CMAKE_PREFIX_PATH}
        )

# packages
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development NumPy)
find_package(Torch REQUIRED)

# sources
set(GUI
        include/gui/SketchBoard.h
        include/gui/ScribbleArea.h
        src/gui/SketchBoard.cpp
        src/gui/ScribbleArea.cpp
        )

set(MATPLOTLIB-CPP
        include/matplotlib-cpp/matplotlibcpp.h
        )

set(MAIN
        src/main.cpp
        )

set(TBB_LIBS
        /usr/lib/x86_64-linux-gnu/libtbb.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
        )

set(ALL_SOURCE_FILES
        ${GUI}
        ${MAIN}
        ${MATPLOTLIB-CPP}
        )

set(ALL_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
        include
        )

message(${OpenCV_LIBS})

set(ALL_LIBRARIES
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${TBB_LIBS}
        ${TORCH_LIBRARIES}
        Eigen3::Eigen
        pthread
        Python3::Python
        Python3::NumPy
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        )

set(ALL_COMPILE_DEFS
        -DQT_NO_KEYWORDS
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

