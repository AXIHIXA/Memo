cmake_minimum_required(VERSION 3.16)
project(OpenGLDemo3D)
set(CMAKE_CXX_STANDARD 17)

find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# all sources

set(APP
        include/app/App.h
        include/app/Window.h
        src/app/App.cpp
        src/app/Window.cpp
)

set(GLAD
        include/glad/glad.h
        include/KHR/khrplatform.h
        src/glad/glad.c
)

set(UTIL
        include/util/Camera.h
        include/util/Shader.h
)

set(SHAPE
        include/shape/Line.h
        include/shape/Mesh.h
        include/shape/Shape.h
        include/shape/Sphere.h
        include/shape/Tetrahedron.h
        src/shape/Line.cpp
        src/shape/Mesh.cpp
        src/shape/Shape.cpp
        src/shape/Sphere.cpp
        src/shape/Tetrahedron.cpp
)

set(ALL_INCLUDE_DIRS
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${OPENGL_EGL_INCLUDE_DIRS}
        include
)

set(ALL_SOURCE_FILES
        ${APP}
        ${GLAD}
        ${SHAPE}
        ${UTIL}
        src/main.cpp
)

set(ALL_LIBRARIES
        dl
        GLEW::GLEW
        glfw
        OpenGL::GL
        OpenGL::OpenGL
        OpenGL::GLU
        OpenGL::GLX
        OpenGL::EGL
        pthread
)

# all definitions and options

set(EXECUTABLE ${PROJECT_NAME})

set(ALL_COMPILE_DEFS
        -DWINDOW_NAME="${EXECUTABLE}"
)

set(ALL_COMPILE_OPTS
        -Wpessimizing-move
        -Wredundant-move
)

# executable target(s)

add_executable(${EXECUTABLE} ${ALL_SOURCE_FILES})
target_compile_definitions(${EXECUTABLE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${EXECUTABLE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${EXECUTABLE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} ${ALL_LIBRARIES})
