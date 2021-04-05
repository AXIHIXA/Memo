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
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development Interpreter NumPy)

# CMAKE_PREFIX_PATH can be gotten from `python -m "import torch;print(torch.utils.cmake_prefix_path)"`
# libTorch conda build conflicts with OpenCV, so download compiled library directly from pytorch.org.
find_package(Torch REQUIRED CONFIG HINTS /home/xihan1/lib/libtorch/share/cmake)

set(GLOBAL
        include/global/global.h
        include/global/Style.h
        src/global/global.cpp
        )

set(GUI
        include/gui/SketchBoard.h
        include/gui/ScribbleArea.h
        include/gui/Stroke.h
        src/gui/SketchBoard.cpp
        src/gui/ScribbleArea.cpp
        src/gui/Stroke.cpp
        )

set(MAIN
        src/main/main.cpp
        )

set(MODULE
        include/module/LaplacianNet.h
        include/module/SolverNet.h
        src/module/LaplacianNet.cpp
        src/module/SolverNet.cpp
        )

set(UTILS
        include/utils/helper.h
        include/utils/matplotlibcpp.h
        src/utils/helper.cpp
        )

set(TBB_LIBS
        /usr/lib/x86_64-linux-gnu/libtbb.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
        )

set(ALL_SOURCE_FILES
        ${GLOBAL}
        ${GUI}
        ${MAIN}
        ${MODULE}
        ${UTILS}
        )

set(ALL_INCLUDE_DIRS
        ${OpenCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${OpenMP_CXX_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
        /home/$ENV{USER}/lib/amgcl
        include
        )

set(ALL_LIBRARIES
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${OpenCL_LIBRARIES}
        ${TBB_LIBS}
        ${TORCH_LIBRARIES}
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
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

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${ALL_SOURCE_FILES})
target_compile_definitions(${EXECUTABLE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${EXECUTABLE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${EXECUTABLE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})

