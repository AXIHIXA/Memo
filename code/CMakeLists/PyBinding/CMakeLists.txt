cmake_minimum_required(VERSION 3.27)
project(CPX CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt-10.0.0/build/")

set(Python_ROOT_DIR
        "$ENV{HOME}/opt/anaconda3/envs/py3"
        )
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG HINTS "${Python_ROOT_DIR}/lib/python3.10/site-packages/pybind11/share/cmake/")

set(UTIL
        include/util/add.h
        src/util/add.cpp
        )

set(MAIN
        src/main.cpp
        )

set(ALL_SOURCE_FILES
        ${MAIN}
        ${UTIL}
        )

set(ALL_INCLUDE_DIRS
        ${fmt_INCLUDE_DIRS}
        include
        )

set(ALL_LIBRARIES
        fmt::fmt
        pybind11::module
        pybind11::embed
        )

set(MODULE cpx)

set(ALL_COMPILE_DEFS
        -DMODULE_NAME=${MODULE}
        )

set(ALL_COMPILE_OPTS
        -Wpessimizing-move
        -Wredundant-move
        )

pybind11_add_module(${MODULE} ${ALL_SOURCE_FILES})
target_compile_definitions(${MODULE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${MODULE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${MODULE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${MODULE} PUBLIC ${ALL_LIBRARIES})
