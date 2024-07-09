cmake_minimum_required(VERSION 3.27)
project(CppDemo)
set(CMAKE_CXX_STANDARD 20)

find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt/build/")

set(UTIL

)

set(ALL_SOURCE_FILES
        ${UTIL}
        ${MAIN}
)

set(ALL_INCLUDE_DIRS
        # ${Boost_INCLUDE_DIRS} points to anaconda3 headers under conda environment (defaulted by CLion),
        # shadowing libfmt's search path!
        # This wrong header version will conflict with libfmt.a!
        include
)

set(ALL_LIBRARIES
#        $<$<CONFIG:Debug>:asan>
        fmt::fmt
        tbb
)

set(ALL_COMPILE_DEFS

)

set(ALL_COMPILE_OPTS
#        $<$<CONFIG:Debug>:-fsanitize=address>
        # -g is included in CMAKE_CXX_FLAGS_DEBUG by default.
        $<$<CONFIG:Debug>:-O0>
        -Wpessimizing-move
        -Wredundant-move
)

set(DEMO demo)
add_executable(${DEMO} ${ALL_SOURCE_FILES} src/main.cpp)
target_compile_definitions(${DEMO} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${DEMO} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${DEMO} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${DEMO} ${ALL_LIBRARIES})

set(LUOGU luogu)
add_executable(${LUOGU} ${ALL_SOURCE_FILES} src/luogu.cpp)
target_compile_definitions(${LUOGU} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${LUOGU} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${LUOGU} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${LUOGU} ${ALL_LIBRARIES})
