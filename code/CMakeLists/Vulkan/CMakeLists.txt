cmake_minimum_required(VERSION 3.24)
project(VulkanDemo)
set(CMAKE_CXX_STANDARD 20)

find_package(Boost REQUIRED)
find_package(glfw3 REQUIRED)
find_package(fmt REQUIRED HINTS "$ENV{HOME}/lib/fmt/build/")
find_package(OpenCV REQUIRED)
find_package(Vulkan REQUIRED COMPONENTS glslc)

# Compile GLSL shader sources into SPIR-V format with CMake
# Reference: https://www.reddit.com/r/vulkan/comments/kbaxlz/what_is_your_workflow_when_compiling_shader_files/
find_program(GLSLC NAMES glslc HINTS Vulkan::glslc)

function(target_spirv_shader TARGET)
    cmake_parse_arguments(SHADER "" "" "VERT;TESC;TESE;GEOM;FRAG" ${ARGN})
    set(OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/shader/)
    # file(REMOVE_RECURSE ${OUTPUT_DIRECTORY})
    file(MAKE_DIRECTORY ${OUTPUT_DIRECTORY})
    foreach (STAGE VERT TESC TESE GEOM FRAG)
        foreach (${STAGE}_GLSL_PATH IN LISTS SHADER_${STAGE})
            get_filename_component(${STAGE}_NAME ${${STAGE}_GLSL_PATH} NAME_WLE)
            set(${STAGE}_SPV_PATH ${OUTPUT_DIRECTORY}/${${STAGE}_NAME}.spv)
            string(TOLOWER ${STAGE} STAGE_LOWER)
            add_custom_command(
                    OUTPUT ${STAGE}_SPV_PATH
                    COMMAND ${GLSLC} -fshader-stage=${STAGE_LOWER} ${${STAGE}_GLSL_PATH} -o ${${STAGE}_SPV_PATH}
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    DEPENDS ${${STAGE}_GLSL_PATH}
                    IMPLICIT_DEPENDS CXX ${${STAGE}_GLSL_PATH}
                    VERBATIM)
            set_source_files_properties(${${STAGE}_SPV_PATH} PROPERTIES GENERATED TRUE)
            target_sources(${TARGET} PRIVATE ${STAGE}_SPV_PATH)
        endforeach ()
    endforeach ()
endfunction()

set(APP
        include/app/VulkanApp.h
        src/app/VulkanApp.cpp
        )

set(MAIN
        src/main.cpp
        )

set(TBB_LIBRARIES
        # Do NOT include libtbbmalloc and libtbbmaloc_proxy, otherwise crashes after exiting Python intepreter
        /usr/lib/x86_64-linux-gnu/libtbb.so
        )

set(ALL_SOURCE_FILES
        ${APP}
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Vulkan_INCLUDE_DIRS}
        include
        )

set(ALL_LIBRARIES
        ${OpenCV_LIBS}
        ${TBB_LIBRARIES}
        dl
        fmt::fmt
        glfw
        pthread
        Vulkan::Vulkan
        X11
        Xi
        Xrandr
        Xxf86vm
        )

set(ALL_COMPILE_DEFS
        -DWINDOW_NAME="VulkanDemo"
        -DGLFW_INCLUDE_VULKAN
        ${ALL_COMPILE_DEFS}
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
target_spirv_shader(${EXECUTABLE} VERT src/shader/main.vert.glsl FRAG src/shader/main.frag.glsl)

