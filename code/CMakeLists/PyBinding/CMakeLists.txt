cmake_minimum_required(VERSION 3.27)
project(CPX CXX)
set(CMAKE_CXX_STANDARD 17)

# If the main consumer of libfmt is a shared library (i.e., a Python extension),
# we must build libfmt as either shared library or position-independent static library.
# See https://fmt.dev/latest/usage.html for details.
find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt-10.0.0/build/")

set(Python_ROOT_DIR
        "$ENV{HOME}/opt/anaconda3/envs/py3/"
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

set(PYMODULE cpx)

set(ALL_COMPILE_DEFS
        -DMODULE_NAME=${PYMODULE}
        )

set(ALL_COMPILE_OPTS
        -Wpessimizing-move
        -Wredundant-move
        )

pybind11_add_module(${PYMODULE} ${ALL_SOURCE_FILES})
target_compile_definitions(${PYMODULE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${PYMODULE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${PYMODULE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PYMODULE} PUBLIC ${ALL_LIBRARIES})
