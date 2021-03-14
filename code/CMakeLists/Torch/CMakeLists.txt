cmake_minimum_required(VERSION 3.16)
project(SketchBoard)
set(CMAKE_CXX_STANDARD 17)

# These 3 options for Qt support
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(Python3 REQUIRED COMPONENTS Development NumPy)

# CMAKE_PREFIX_PATH can be gotten from `python -m "import torch;print(torch.utils.cmake_prefix_path)"`
# libTorch conda build conflicts with OpenCV, so download compiled library directly from pytorch.org.
find_package(Torch REQUIRED CONFIG HINTS /home/xihan1/lib/libtorch/share/cmake)

set(GUI
        include/gui/SketchBoard.h
        include/gui/ScribbleArea.h
        src/gui/SketchBoard.cpp
        src/gui/ScribbleArea.cpp
        )

set(NET
        include/net/UNet.h
        include/net/UNetParts.h
        src/net/UNet.cpp
        src/net/UNetParts.cpp
        )

set(UTILS
        include/utils/matplotlibcpp.h
        include/utils/tensorhelper.h
        src/utils/tensorhelper.cpp
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
        ${NET}
        ${UTILS}
        )

set(ALL_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
        include
        )

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

