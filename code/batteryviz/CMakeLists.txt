cmake_minimum_required(VERSION 3.10)

project(batteryviz LANGUAGES CXX CUDA)

find_library(CUDART_LIBRARY cudart 
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} $ENV{CUDA_PATH}\\lib\\x64
)

find_library(CUSPARSE_LIBRARY cusparse 
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} $ENV{CUDA_PATH}\\lib\\x64
)

if(WIN32)
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES $ENV{CUDA_PATH}\\include)
endif()

#GLFW 
if (NOT WIN32)
	find_package(glfw3 REQUIRED)
else()
	find_library(GLFW_LIBRARY glfw)   
endif()
message(${GLFW3_INCLUDE_DIRS})
include_directories(${GLFW3_INCLUDE_DIRS})


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
set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set (IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
 
set(_SRCS 
    ${PROJECT_SOURCE_DIR}/App.cpp
    ${PROJECT_SOURCE_DIR}/BatteryApp.cpp
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/Ui.cpp
    ${PROJECT_SOURCE_DIR}/render/Camera.cpp
    ${PROJECT_SOURCE_DIR}/render/Framebuffer.cpp
    ${PROJECT_SOURCE_DIR}/render/GLRenderer.cpp
    ${PROJECT_SOURCE_DIR}/render/Light.cpp
    ${PROJECT_SOURCE_DIR}/render/MeshObject.cpp
    ${PROJECT_SOURCE_DIR}/render/PrimitivesVBO.cpp
    ${PROJECT_SOURCE_DIR}/render/RenderList.cpp
    ${PROJECT_SOURCE_DIR}/render/Scene.cpp
    ${PROJECT_SOURCE_DIR}/render/Shader.cpp
    ${PROJECT_SOURCE_DIR}/render/ShaderResource.cpp
    ${PROJECT_SOURCE_DIR}/render/Shaders.cpp
    ${PROJECT_SOURCE_DIR}/render/Texture.cpp
    ${PROJECT_SOURCE_DIR}/render/VertexBuffer.cpp
    ${PROJECT_SOURCE_DIR}/render/VolumeRaycaster.cpp
    ${PROJECT_SOURCE_DIR}/utility/IOUtility.cpp
    ${PROJECT_SOURCE_DIR}/utility/Options.cpp  
    #external
    ${PROJECT_EXTERNAL_DIR}/imgui/imgui.cpp
    ${PROJECT_EXTERNAL_DIR}/imgui/imgui_draw.cpp
    ${PROJECT_EXTERNAL_DIR}/imgui/imgui_file_explorer.cpp
    ${PROJECT_EXTERNAL_DIR}/imgui/imgui_impl_glfw_gl3.cpp
)

set(_HEADERS
	${PROJECT_INCLUDE_DIR}/App.h
    ${PROJECT_INCLUDE_DIR}/BatteryApp.h    
    ${PROJECT_INCLUDE_DIR}/Ui.h
    ${PROJECT_INCLUDE_DIR}/render/Camera.h
    ${PROJECT_INCLUDE_DIR}/render/Framebuffer.h
    ${PROJECT_INCLUDE_DIR}/render/GLRenderer.h
    ${PROJECT_INCLUDE_DIR}/render/Light.h
    ${PROJECT_INCLUDE_DIR}/render/MeshObject.h
    ${PROJECT_INCLUDE_DIR}/render/PrimitivesVBO.h
    ${PROJECT_INCLUDE_DIR}/render/RenderList.h
    ${PROJECT_INCLUDE_DIR}/render/Scene.h
    ${PROJECT_INCLUDE_DIR}/render/Shader.h
    ${PROJECT_INCLUDE_DIR}/render/ShaderResource.h
    ${PROJECT_INCLUDE_DIR}/render/Shaders.h
    ${PROJECT_INCLUDE_DIR}/render/Texture.h
    ${PROJECT_INCLUDE_DIR}/render/VertexBuffer.h
    ${PROJECT_INCLUDE_DIR}/render/VolumeRaycaster.h
    ${PROJECT_INCLUDE_DIR}/utility/IOUtility.h
    ${PROJECT_INCLUDE_DIR}/utility/Options.h
)

#Include directories
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_EXTERNAL_DIR}")
include_directories("${IMGUI_DIR}")
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories("${GLM_INCLUDE_DIR}")


#include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
#include_directories("${BATTERYLIB_INCLUDE}")


if (NOT WIN32)
find_package (Eigen3 3.3 REQUIRED)
endif()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )


add_definitions(${batterylib_DEFINITIONS})

add_executable(${PROJECT_NAME} ${_SRCS} ${_HEADERS})

add_dependencies(${PROJECT_NAME} batterylib)
target_link_libraries(${PROJECT_NAME} batterylib)

if(NOT WIN32)
	target_link_libraries(${PROJECT_NAME} glfw)
else()
	target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARY})
endif()


target_link_libraries(${PROJECT_NAME} ${CUDART_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${CUSPARSE_LIBRARY})

if(NOT WIN32)
target_link_libraries(${PROJECT_NAME} omp)
target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()


if(WIN32)
set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()