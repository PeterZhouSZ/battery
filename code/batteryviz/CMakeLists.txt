cmake_minimum_required(VERSION 3.10)

project(batteryviz LANGUAGES CXX)


#Set main directories
set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
 
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
)

#Include directories
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_EXTERNAL_DIR}")

#include_directories("${BATTERYLIB_INCLUDE}")


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )


add_executable(${PROJECT_NAME} ${_SRCS})

add_dependencies(${PROJECT_NAME} batterylib)
target_link_libraries(${PROJECT_NAME} batterylib)