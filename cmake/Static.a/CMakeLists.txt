cmake_minimum_required(VERSION 3.0)
project(hw5_clion C)
set(CMAKE_C_STANDARD 99)

# 1, ON, YES, TRUE, Y
# 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, empty string, <string>-NOTFOUND
set(TEST OFF)

set(SOURCE_FILES
        include/client_registry.h
        include/debug.h
        include/exchange.h
        include/protocol.h
        include/server.h
        include/trader.h
        )

set(ALL_LIBS
        ${CMAKE_SOURCE_DIR}/lib/bourse.a
        ${CMAKE_SOURCE_DIR}/lib/bourse_debug.a
        criterion
        pthread
        )

if (${TEST})
    set(SOURCE_FILES
            ${SOURCE_FILES}
            tests/bourse_tests.c
            )
else ()
    set(SOURCE_FILES
            ${SOURCE_FILES}
            src/main.c
            )
endif ()

add_definitions(-DCOLOR -DVERBOSE)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "include")
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})