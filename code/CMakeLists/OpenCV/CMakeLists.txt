cmake_minimum_required(VERSION 3.16)
project(OpenCVDemo)
set(CMAKE_CXX_STANDARD 17)

# packages to find

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# all sources

set(MAIN
        src/main/main.cpp
        )

set(ALL_SOURCE_FILES
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        include
        )

set(ALL_LIBRARIES
        ${OpenCV_LIBS}
        )

# executable target

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_compile_options(${PROJECT_NAME} PUBLIC -Wpessimizing-move -Wredundant-move)
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBRARIES})
