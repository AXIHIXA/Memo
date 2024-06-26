cmake_minimum_required(VERSION 3.27)
project(CppDemo)
set(CMAKE_CXX_STANDARD 20)

find_package(Boost REQUIRED)
find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt/build/")

set(UTIL

)

set(MAIN
        src/main.cpp
)

set(ALL_SOURCE_FILES
        ${UTIL}
        ${MAIN}
)

set(ALL_INCLUDE_DIRS
        ${Boost_INCLUDE_DIRS}
        include
)

set(ALL_LIBRARIES
        $<$<CONFIG:Debug>:asan>
        fmt::fmt
        tbb
)

set(ALL_COMPILE_DEFS
        
)

set(ALL_COMPILE_OPTS
        $<$<CONFIG:Debug>:-fsanitize=address>
        # $<$<CONFIG:Debug>:-g>  # Already included in CMAKE_CXX_FLAGS_DEBUG. 
        $<$<CONFIG:Debug>:-O1>
        -Wpessimizing-move
        -Wredundant-move
)

set(EXECUTABLE demo)
add_executable(${EXECUTABLE} ${ALL_SOURCE_FILES})
target_compile_definitions(${EXECUTABLE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${EXECUTABLE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${EXECUTABLE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})
