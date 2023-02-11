cmake_minimum_required(VERSION 3.16)
project(CppDemo)
set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)
find_package(fmt REQUIRED HINTS "/home/xihan1/lib/fmt-9.1.0/")
find_package(OpenCV REQUIRED)

set(UTIL
        include/util/TimerGuard.h
        src/util/TimerGuard.cpp
        )

set(MAIN
        include/S35.h
        src/main.cpp
        )

set(TBB_LIBRARIES
        /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
        /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
        /usr/lib/x86_64-linux-gnu/libtbb.so
        )

set(ALL_SOURCE_FILES
        ${UTIL}
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        ${Boost_INCLUDE_DIRS}
        ${FMT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        include
        )

set(ALL_LIBRARIES
        fmt::fmt
        ${OpenCV_LIBS}
        ${TBB_LIBRARIES}
        )

set(ALL_COMPILE_DEFS

        )

set(ALL_COMPILE_OPTS
        -Wpessimizing-move
        -Wredundant-move
        )

set(EXECUTABLE ${PROJECT_NAME})
add_executable(${EXECUTABLE} ${ALL_SOURCE_FILES})
target_compile_definitions(${EXECUTABLE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${EXECUTABLE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${EXECUTABLE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})

