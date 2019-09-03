cmake_minimum_required(VERSION 3.13)
project(hw1 C)

set(CMAKE_C_STANDARD 99)

include_directories("${PROJECT_SOURCE_DIR}/include/")

set(SOURCES
        src/transplant.c
        src/main.c)

add_executable(hw1 ${SOURCES})