cmake_minimum_required(VERSION 3.0)
project(Demo)
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        include/debug.h
        src/main.cpp
        )

set(ALL_INCLUDE_DIRS
        include
        )

set(ALL_LIBS
        )

add_definitions(-DCOLOR -DVERBOSE)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})
