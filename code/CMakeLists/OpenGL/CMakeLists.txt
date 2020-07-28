cmake_minimum_required(VERSION 3.16)
project(OpenGLDemo)
set(CMAKE_CXX_STANDARD 17)

# packages to find

find_package(CGAL REQUIRED)
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

# OpenMesh

set(OPENMESH_INCLUDE_DIRS
        /home/ax/lib/OpenMesh/include
        )

set(OPENMESH_LIBRARIES
        /home/ax/lib/OpenMesh/lib/libOpenMeshCore.so
        /home/ax/lib/OpenMesh/lib/libOpenMeshTools.so
        )

# all sources

set(MAIN
        src/main/main.cpp
        )

set(ALL_SOURCE_FILES
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        ${GLEW_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${OPENMESH_INCLUDE_DIRS}
        include
        )

set(ALL_LIBRARIES
        ${OpenCV_LIBS}
        ${OPENMESH_LIBRARIES}
        CGAL::CGAL
        dl
        Eigen3::Eigen
        GLEW::GLEW
        glfw
        OpenGL::GL
        OpenGL::OpenGL
        OpenGL::GLU
        OpenGL::GLX
        OpenGL::EGL
        pthread
        )

# executable target

add_executable(${PROJECT_NAME} ${ALL_SOURCE_FILES})
target_compile_options(${PROJECT_NAME} PUBLIC -Wpessimizing-move -Wredundant-move)
target_include_directories(${PROJECT_NAME} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBRARIES})
