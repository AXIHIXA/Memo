cmake_minimum_required(VERSION 3.0)
project(BoostDemo CXX)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_PREFIX_PATH
        /usr/lib/cmake/eigen3/
        ${CMAKE_PREFIX_PATH})

find_package(Eigen3 REQUIRED)

set(SOURCE_FILES
        src/main.cpp)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${EIGEN3_LIBRARY})
