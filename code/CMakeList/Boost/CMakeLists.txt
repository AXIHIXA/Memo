cmake_minimum_required(VERSION 3.10)
project(CppPrimer)
set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)

set(SOURCE_FILES
        src/main.cpp
        )

set(ALL_INCLUDE_DIRS
        include
        ${Boost_INCLUDE_DIRS}
        )

set(ALL_LIBS
        ${Boost_LIBRARY_DIRS}
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})
