cmake_minimum_required(VERSION 3.10)
#SET (CMAKE_C_COMPILER "/usr/bin/clang")
#SET (CMAKE_CXX_COMPILER "/usr/bin/clang++")

project(Battery)


if (WIN32)

	#Set GLEW path from GLEW_PATH env. variable
	list(APPEND CMAKE_PREFIX_PATH $ENV{GLEW_PATH})
	list(APPEND CMAKE_LIBRARY_PATH $ENV{GLEW_PATH}\\lib\\Release\\x64)

	#Set GLFW3 path from GLFW_PATH env. variable
	set(GLFW3_INCLUDE_DIRS $ENV{GLFW_PATH}\\include)
	list(APPEND CMAKE_LIBRARY_PATH $ENV{GLFW_PATH}\\src\\Release\\)	

	#Set EIGEN3 path from EIGEN_PATH env. variable
	set(EIGEN3_INCLUDE_DIR $ENV{EIGEN_PATH})

	#Set Cuda runtime path from CUDA_PATH
	
	
	#list(APPEND CMAKE_PREFIX_PATH $ENV{CUDA_PATH})
	
	set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES $ENV{CUDA_PATH}\\include)

	set(GLM_INCLUDE_DIR $ENV{GLM_PATH})

	#message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

	#list(APPEND CMAKE_PREFIX_PATH $ENV{EIGEN_PATH}/)

	#message("EIGEN test path: " "$ENV{EIGEN_PATH}\\cmake\\")

	#set(Eigen3_DIR )
else()
	set(GLM_INCLUDE_DIR "${PROJECT_INCLUDE_DIR}")
endif()




set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build) 
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}) 
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif ()


if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE   ON)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17")


if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /std:c++latest /openmp")	 
endif()






add_subdirectory(batterylib)
add_subdirectory(batteryviz)
add_subdirectory(batterytool)

