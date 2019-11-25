cmake_minimum_required(VERSION 3.0)
project(hw4_raw C)
set(CMAKE_C_STANDARD 99)

# 1, ON, YES, TRUE, Y
# 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, empty string, <string>-NOTFOUND
set(TEST OFF)

set(SOURCE_FILES
        include/debug.h
        include/global.h
        include/handler.h
        include/jobber.h
        include/sf_readline.h
        include/task.h
        src/handler.c
        src/jobs.c
        src/sf_readline.c
        src/task.c
)

SET(OBJECT_FILES
        lib/sf_event.o
)

if (${TEST})
    set(SOURCE_FILES
            ${SOURCE_FILES}
            tests/hw4_tests.c
    )
else ()
    set(SOURCE_FILES
            ${SOURCE_FILES}
            src/main.c
    )
endif ()

add_definitions(-DCOLOR -DVERBOSE)
add_compile_options(-D_POSIX_SOURCE -D_DEFAULT_SOURCE)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${OBJECT_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "include")
target_link_libraries(${PROJECT_NAME} criterion)
