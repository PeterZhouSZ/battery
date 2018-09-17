cmake_minimum_required(VERSION 3.10)

project(batterytool LANGUAGES CXX CUDA)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUSPARSE_LIBRARY cusparse ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})


#Opengl
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
link_libraries(${OPENGL_LIBRARIES})

#Glew 
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_LIBRARIES})



#Set main directories
set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
 
set(_SRCS 
    ${PROJECT_SOURCE_DIR}/main.cpp    
)

#Include directories
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_EXTERNAL_DIR}")
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")



find_package (Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )


#add_definitions(${batterylib_DEFINITIONS})

add_executable(${PROJECT_NAME} ${_SRCS})

add_dependencies(${PROJECT_NAME} batterylib)
target_link_libraries(${PROJECT_NAME} batterylib)


target_link_libraries(${PROJECT_NAME} omp)
target_link_libraries(${PROJECT_NAME} stdc++fs)
target_link_libraries(${PROJECT_NAME} ${CUSPARSE_LIBRARY})
