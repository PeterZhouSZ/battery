cmake_minimum_required(VERSION 3.10)
SET (CMAKE_C_COMPILER "/usr/bin/clang")
SET (CMAKE_CXX_COMPILER "/usr/bin/clang++")

project(Battery)





set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build) 
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}) 
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")


set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -std=libc++")

if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")
endif()


add_subdirectory(batterylib)
add_subdirectory(batteryviz)
#add_subdirectory(batterytool)

