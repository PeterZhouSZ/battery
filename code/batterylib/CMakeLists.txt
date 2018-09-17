cmake_minimum_required(VERSION 3.10)

project(batterylib LANGUAGES CXX CUDA)

find_library(CUDART_LIBRARY cudart 
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} $ENV{CUDA_PATH}\\lib\\x64
)

find_library(CUSPARSE_LIBRARY cusparse 
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} $ENV{CUDA_PATH}\\lib\\x64
)

if(WIN32)
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES $ENV{CUDA_PATH}\\include)
endif()



#Opengl
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
link_libraries(${OPENGL_LIBRARIES})

#Glew 
find_package(GLEW REQUIRED)
if(GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif()



#Set main directories
set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set (BATTERYLIB_INCLUDE PROJECT_INCLUDE_DIR)
set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set (TINYTIFF_DIR ${PROJECT_EXTERNAL_DIR}/tinytiff)
 
set(_SRCS 
    ${PROJECT_SOURCE_DIR}/AABB.cpp
    ${PROJECT_SOURCE_DIR}/DataPtr.cpp
    ${PROJECT_SOURCE_DIR}/DiffusionSolver.cpp
    ${PROJECT_SOURCE_DIR}/Ellipsoid.cpp
    ${PROJECT_SOURCE_DIR}/GLGlobal.cpp
    ${PROJECT_SOURCE_DIR}/MGGPU.cpp
    ${PROJECT_SOURCE_DIR}/MultigridGPU.cpp
    ${PROJECT_SOURCE_DIR}/MultigridSolver.cpp
    ${PROJECT_SOURCE_DIR}/OrientationHistogram.cpp
    ${PROJECT_SOURCE_DIR}/RandomGenerator.cpp
    ${PROJECT_SOURCE_DIR}/SimulatedAnnealing.cpp
    ${PROJECT_SOURCE_DIR}/Timer.cpp
    ${PROJECT_SOURCE_DIR}/Transform.cpp
    ${PROJECT_SOURCE_DIR}/Triangle.cpp
    ${PROJECT_SOURCE_DIR}/TriangleMesh.cpp
    ${PROJECT_SOURCE_DIR}/Types.cpp
    ${PROJECT_SOURCE_DIR}/Volume.cpp
    ${PROJECT_SOURCE_DIR}/VolumeIO.cpp
    ${PROJECT_SOURCE_DIR}/cuda/CudaMath.h
    ${PROJECT_SOURCE_DIR}/cuda/CudaUtility.cpp    
    ${PROJECT_SOURCE_DIR}/cuda/MGGPU.cu    
    ${PROJECT_SOURCE_DIR}/cuda/MultigridGPU.cu    
    ${PROJECT_SOURCE_DIR}/cuda/MultigridGPUNew.cu    
    ${PROJECT_SOURCE_DIR}/cuda/Volume.cu    
    ${TINYTIFF_DIR}/tinytiffreader.cpp
)



#Include directories
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_EXTERNAL_DIR}")
include_directories(${PROJECT_SOURCE_DIR}/cuda)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories("${GLM_INCLUDE_DIR}")

include_directories(../batterylib/include)

if (NOT WIN32)
find_package (Eigen3 3.3 REQUIRED)
endif()

INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

#Output external dependencies
#message(STATUS "TinyTiff directory: " ${TINYTIFF_DIR})
#message(STATUS "CUDA RT directory: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#message(STATUS "EIGEN directory: " ${EIGEN3_INCLUDE_DIR})


add_library(${PROJECT_NAME} SHARED ${_SRCS})

#CUDA options
#target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_definitions(-DBATTERYLIB_EXPORT)

target_link_libraries(${PROJECT_NAME} ${CUDART_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${CUSPARSE_LIBRARY})


set_target_properties( ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w" )                       
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )   
set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -std=c++11 -w")

set(${PROJECT_NAME}_DEFINITIONS CACHE INTERNAL "${PROJECT_NAME}: Definitions" FORCE)



