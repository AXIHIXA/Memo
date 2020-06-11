cmake_minimum_required(VERSION 3.16)
project(BoostFSDemo)
set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS filesystem)

aux_source_directory(main/src MAIN)

set(ALL_SOURCE_FILES
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        main/include
        )

set(ALL_LIBRARIES
        Boost::filesystem
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBRARIES})
