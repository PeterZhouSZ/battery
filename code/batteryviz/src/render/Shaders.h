#pragma once

#include <array>
#include <unordered_map>
#include <memory>

struct Shader;

enum ShaderType  {
	SHADER_PHONG,
	SHADER_POSITION,
	SHADER_VOLUME_RAYCASTER,
	SHADER_VOLUME_SLICE,
	SHADER_COUNT
};


using ShaderDB = std::array<
	std::shared_ptr<Shader>, 
	ShaderType::SHADER_COUNT
>;


#define SHADER_PATH "../batteryviz/src/shaders/"

const std::array<
	const char *,
	ShaderType::SHADER_COUNT
> g_shaderPaths = {
	"forwardphong",
	"position",
	"volumeraycast",
	"volumeslice"
};