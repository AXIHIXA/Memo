cmake_minimum_required(VERSION 3.16)
project(OpenCVDemo)
set(CMAKE_CXX_STANDARD 17)

# packages to find

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

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
        include
        )

set(ALL_LIBRARIES
        ${OpenCV_LIBS}
        )
        
# all definitions and options

set(ALL_COMPILE_DEFS

        )

set(ALL_COMPILE_OPTS
        -Wpessimizing-move
        -Wredundant-move
        )

# executable target

add_executable(${PROJECT_NAME} ${ALL_SOURCE_FILES})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${PROJECT_NAME} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBRARIES})
