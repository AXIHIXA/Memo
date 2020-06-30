cmake_minimum_required(VERSION 3.16)
project(CppDemo)
set(CMAKE_CXX_STANDARD 17)

aux_source_directory(src/main/src MAIN)

set(ALL_SOURCE_FILES
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        src/main/include
        )

set(ALL_LIBRARIES

        )

add_executable(${PROJECT_NAME} ${ALL_SOURCE_FILES})
target_compile_options(${PROJECT_NAME} PUBLIC -Wpessimizing-move -Wredundant-move)
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBRARIES})