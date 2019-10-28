cmake_minimum_required(VERSION 3.13)

set(PROJECT_NAME hw4_raw_cl)
set(CMAKE_C_STANDARD 99)
# 1, ON, YES, TRUE, Y
# 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, empty string, <string>-NOTFOUND
set(TEST OFF)

add_definitions(-DDEBUG -DCOLOR -DERROR -DSUCCESS -DWARN -DINFO)
#add_compile_options(-Wall -Werror -Wno-unused-function -MMD)
#add_compile_options(-D_POSIX_SOURCE -D_DEFAULT_SOURCE)

include_directories("include")

set(SOURCE_FILES
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

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${OBJECT_FILES})

target_link_libraries(${PROJECT_NAME} criterion)
