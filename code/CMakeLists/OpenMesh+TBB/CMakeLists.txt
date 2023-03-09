cmake_minimum_required(VERSION 3.16)
project(OpenMeshDemo)
set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)
find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# OpenMesh

set(OPENMESH_INCLUDE_DIRS
        $ENV{HOME}/lib/OpenMesh/include
        )

set(OPENMESH_LIBRARIES
        $ENV{HOME}/lib/OpenMesh/lib/libOpenMeshCore.so
        $ENV{HOME}/lib/OpenMesh/lib/libOpenMeshTools.so
        )

# TBB

set(TBB_LIBRARIES
        /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
        /usr/lib/x86_64-linux-gnu/libtbb.so
        )

# all sources

set(MAIN
        src/main.cpp
        )

set(ALL_SOURCE_FILES
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${OPENMESH_INCLUDE_DIRS}
        include
        )

set(ALL_LIBRARIES
        ${OpenCV_LIBS}
        ${OPENMESH_LIBRARIES}
        ${TBB_LIBRARIES}
        CGAL::CGAL
        )

# all definitions and options

set(ALL_COMPILE_DEFS
        -DVISUALIZE_DEPTH_MAP
        )

set(ALL_COMPILE_OPTS
        -Wpessimizing-move
        -Wredundant-move
        )

# executable target

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${ALL_SOURCE_FILES})
target_compile_definitions(${EXECUTABLE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${EXECUTABLE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${EXECUTABLE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})
