cmake_minimum_required(VERSION 3.25)
project(CppDemo CXX)
set(CMAKE_CXX_STANDARD 17)

set(DEMO "demo")
add_executable(${DEMO} "src/main.cpp")
target_include_directories(${DEMO} PRIVATE "./include/")
target_compile_options(${DEMO} PRIVATE "$<$<CONFIG:Debug>:-O0>" -fsanitize=undefined)
target_link_libraries(${DEMO} PRIVATE pthread "$<$<CONFIG:Debug>:ubsan>")
target_link_options(${DEMO} PRIVATE)
