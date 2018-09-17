cmake_minimum_required(VERSION 3.10)

project(batteryviz LANGUAGES CXX CUDA)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUSPARSE_LIBRARY cusparse ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})


#GLFW 
find_package(glfw3 REQUIRED)
include_directories(${GLFW3_INCLUDE_DIRS})


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

#Include directories
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_EXTERNAL_DIR}")
include_directories("${IMGUI_DIR}")
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")


#include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
#include_directories("${BATTERYLIB_INCLUDE}")


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )


#add_definitions(${batterylib_DEFINITIONS})

add_executable(${PROJECT_NAME} ${_SRCS})

add_dependencies(${PROJECT_NAME} batterylib)
target_link_libraries(${PROJECT_NAME} batterylib)

target_link_libraries(${PROJECT_NAME} glfw)
target_link_libraries(${PROJECT_NAME} omp)
target_link_libraries(${PROJECT_NAME} stdc++fs)
target_link_libraries(${PROJECT_NAME} ${CUSPARSE_LIBRARY})
