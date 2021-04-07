cmake_minimum_required(VERSION 3.16)
project(Qt5Demo)
set(CMAKE_CXX_STANDARD 17)

# These 3 options for Qt support
# You need to add your header files in add_executable, otherwise the moc won't parse them
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)

set(GUI
        include/gui/MainWindow.h
        src/gui/MainWindow.cpp
        )

set(MAIN
        src/main/main.cpp
        )

set(ALL_SOURCE_FILES
        ${GUI}
        ${MAIN}
        )

set(ALL_INCLUDE_DIRS
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        include
        )

set(ALL_LIBRARIES
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        )

# Disable Qt's bad marco usage to avoid conflicts! 
# After this, call original keywords such as: slots -> Q_SLOTS
set(ALL_COMPILE_DEFS
        -DQT_NO_KEYWORDS
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

