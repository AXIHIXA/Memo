cmake_minimum_required(VERSION 3.0)
project(PVG)
set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

set(SOURCE_FILES
        include/debug.h
        include/global.h
        src/main.cpp
        )

set(ALL_INCLUDE_DIRS
        include
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIR}
        ${OPENGL_INCLUDE_DIR}
        )

set(ALL_LIBS
        ${GLUT_LIBRARY}
        ${OPENGL_LIBRARY}
        ${OpenCV_LIBS}
        

add_definitions(-DST_COLOR -DST_VERBOSE)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})
