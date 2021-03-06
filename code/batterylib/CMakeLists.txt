cmake_minimum_required(VERSION 3.8)

project(batterylib LANGUAGES CXX CUDA)

#Cuda paths
set(CUDA_PATH $ENV{CUDA_PATH})
string(REPLACE "\\" "/" CUDA_PATH ${CUDA_PATH})
find_library(CUDART_LIBRARY cudart 
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} ${CUDA_PATH}\\lib\\x64
)
find_library(CUSPARSE_LIBRARY cusparse 
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} ${CUDA_PATH}\\lib\\x64
)
if(WIN32)
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CUDA_PATH}\\include)
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


file(GLOB_RECURSE SRC_CPP
    "${PROJECT_SOURCE_DIR}/*.cpp"    
)
file(GLOB_RECURSE SRC_H
    "${PROJECT_SOURCE_DIR}/*.h"    
    "${PROJECT_INCLUDE_DIR}/*.h"
)

file(GLOB_RECURSE SRC_EXTERNAL
    "${PROJECT_EXTERNAL_DIR}/*.cpp"    
    "${PROJECT_EXTERNAL_DIR}/*.h"
)

file(GLOB_RECURSE SRC_CUDA_CU
    "${PROJECT_SOURCE_DIR}/*.cu"    
)
file(GLOB_RECURSE SRC_CUDA_CUH
    "${PROJECT_SOURCE_DIR}/*.cuh"    
    "${PROJECT_SOURCE_DIR}/cuda/*.h"
)

#[[set(_SRCS 
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
    
    ${TINYTIFF_DIR}/tinytiffreader.cpp

    ${PROJECT_SOURCE_DIR}/cuda/CudaUtility.cpp    
    ${PROJECT_SOURCE_DIR}/cuda/MGGPU.cu    
    ${PROJECT_SOURCE_DIR}/cuda/MultigridGPU.cu    
    ${PROJECT_SOURCE_DIR}/cuda/MultigridGPUNew.cu    
    ${PROJECT_SOURCE_DIR}/cuda/Volume.cu    
    ${PROJECT_SOURCE_DIR}/cuda/VolumeSurface.cu
    
)

set(_HEADERS 
    ${PROJECT_INCLUDE_DIR}/AABB.h
    ${PROJECT_INCLUDE_DIR}/DataPtr.h
    ${PROJECT_INCLUDE_DIR}/DiffusionSolver.h
    ${PROJECT_INCLUDE_DIR}/Ellipsoid.h
    ${PROJECT_INCLUDE_DIR}/GLGlobal.h
    ${PROJECT_INCLUDE_DIR}/MGGPU.h
    ${PROJECT_INCLUDE_DIR}/MGGPU_Types.h
    ${PROJECT_INCLUDE_DIR}/MultigridGPU.h
    ${PROJECT_INCLUDE_DIR}/MultigridSolver.h
    ${PROJECT_INCLUDE_DIR}/OrientationHistogram.h
    ${PROJECT_INCLUDE_DIR}/RandomGenerator.h
    ${PROJECT_INCLUDE_DIR}/SimulatedAnnealing.h
    ${PROJECT_INCLUDE_DIR}/Timer.h
    ${PROJECT_INCLUDE_DIR}/Transform.h
    ${PROJECT_INCLUDE_DIR}/Triangle.h
    ${PROJECT_INCLUDE_DIR}/TriangleMesh.h
    ${PROJECT_INCLUDE_DIR}/Types.h
    ${PROJECT_INCLUDE_DIR}/Volume.h
    ${PROJECT_INCLUDE_DIR}/VolumeIO.h
    
    ${TINYTIFF_DIR}/tinytiffreader.h

    ${PROJECT_SOURCE_DIR}/cuda/CudaMath.h
    ${PROJECT_SOURCE_DIR}/cuda/CudaUtility.h    
    ${PROJECT_SOURCE_DIR}/cuda/MGGPU.cuh    
    ${PROJECT_SOURCE_DIR}/cuda/MultigridGPU.cuh    
    ${PROJECT_SOURCE_DIR}/cuda/MultigridGPUNew.cuh    
    ${PROJECT_SOURCE_DIR}/cuda/Volume.cuh    
    ${PROJECT_SOURCE_DIR}/cuda/VolumeSurface.cuh
    ${PROJECT_SOURCE_DIR}/cuda/VolumeTypes.cuh

)
]]



#Include directories
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_EXTERNAL_DIR}")
include_directories(${PROJECT_SOURCE_DIR}/cuda)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
if (WIN32)
include_directories("${GLM_INCLUDE_DIR}")
endif()

include_directories(../batterylib/include)

if (NOT WIN32)
find_package (Eigen3 3.3 REQUIRED)
endif()

INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

#Output external dependencies
#message(STATUS "TinyTiff directory: " ${TINYTIFF_DIR})
#message(STATUS "CUDA RT directory: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#message(STATUS "EIGEN directory: " ${EIGEN3_INCLUDE_DIR})


add_library(${PROJECT_NAME} SHARED 
    ${SRC_CPP} ${SRC_H} ${SRC_EXTERNAL} ${SRC_CUDA_CU} ${SRC_CUDA_CUH}
)

source_group("Source"       FILES ${SRC_CPP})
source_group("Headers"      FILES ${SRC_H})
source_group("External"       FILES ${SRC_EXTERNAL})
source_group("CUDA Source"       FILES ${SRC_CUDA_CU})
source_group("CUDA Headers"      FILES ${SRC_CUDA_CUH})


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
if(NOT WIN32)
    set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -std=c++11 -w")
endif()

set(${PROJECT_NAME}_DEFINITIONS CACHE INTERNAL "${PROJECT_NAME}: Definitions" FORCE)



