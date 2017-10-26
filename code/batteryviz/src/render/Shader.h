#pragma once

#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

#include "render/ShaderResource.h"



struct Shader {
	GLuint id;
	std::unordered_map<std::string, ShaderResource> resources;
	ShaderResource & operator [](const std::string & name);
	bool bind();
	static void unbind();
};

bool compileShader(Shader * outputShader,
				   const std::string & code, 
				   std::function<void(const std::string & errorMsg)> errorCallback = {}
);

