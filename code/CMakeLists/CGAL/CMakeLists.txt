cmake_minimum_required(VERSION 3.27)
project(CppDemo)
set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)
find_package(CGAL REQUIRED CONFIG HINTS "$ENV{HOME}/lib/CGAL/build/")
find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt/build/")

set(MAIN
        src/main.cpp
)

set(TBB_LIBRARIES
        # Do NOT include libtbbmalloc and libtbbmaloc_proxy, otherwise crashes after exiting Python intepreter
        /usr/lib/x86_64-linux-gnu/libtbb.so
)

set(ALL_SOURCE_FILES
        ${MAIN}
)

set(ALL_INCLUDE_DIRS
        ${Boost_INCLUDE_DIRS}
        include
)

set(ALL_LIBRARIES
        ${TBB_LIBRARIES}
        CGAL::CGAL
        fmt::fmt
        pthread
        dl
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
