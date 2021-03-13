cmake_minimum_required(VERSION 3.16)
project(Qt5Demo)
set(CMAKE_CXX_STANDARD 17)

# These 3 options for Qt support
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

set(MAIN
        src/main.cpp
        )

set(TBB_LIBS
        /usr/lib/x86_64-linux-gnu/libtbb.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
        )

set(ALL_SOURCE_FILES
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        include
        )

set(ALL_LIBRARIES
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${TBB_LIBS}
        Eigen3::Eigen
        pthread
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
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

