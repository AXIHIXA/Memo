cmake_minimum_required(VERSION 3.13)

set(PROJECT_NAME hw3_cl)
set(CMAKE_C_STANDARD 99)
set(TEST ON)  # 1, ON, YES, TRUE, Y
              # 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, empty string, <string>-NOTFOUND

include_directories("include")

set(SOURCE_FILES
        src/sfmm.c
        src/sfutil.c
)

if (${TEST})
    set(SOURCE_FILES
            ${SOURCE_FILES}
            tests/sfmm_tests.c
    )
else ()
    set(SOURCE_FILES
            ${SOURCE_FILES}
            src/main.c
    )
endif ()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} criterion)
