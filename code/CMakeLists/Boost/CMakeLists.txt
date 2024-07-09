cmake_minimum_required(VERSION 3.16)
project(BoostFSDemo)
set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt/build/")

set(MAIN
        src/main.cpp
)

set(ALL_SOURCE_FILES
        ${MAIN}
)

set(ALL_INCLUDE_DIRS
        include
)

set(ALL_LIBRARIES
        Boost::filesystem
        fmt::fmt
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
