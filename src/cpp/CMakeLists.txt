cmake_minimum_required(VERSION 3.16)
project(OpticalFlowViewer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
#set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/")

# Find Qt5
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Source files
set(SOURCES
    main.cpp
    mainwindow.cpp
    opticalflow.cpp
)

set(HEADERS
    mainwindow.h
    opticalflow.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    Qt5::Core
    Qt5::Widgets
    ${OpenCV_LIBS}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# Compiler warnings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
endif()
