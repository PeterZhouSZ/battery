#include "Shaders.h"

#include <string>
#include "utility/IOUtility.h"


using namespace std;

std::string loadShaders(ShaderDB & targetDB)
{	
	std::string errString;
	for (auto i = 0; i < ShaderType::SHADER_COUNT; i++) {

		const auto path = SHADER_PATH + string(g_shaderPaths[i]) + ".shader";
		auto src = readFileWithIncludes(path);

		if (src.length() == 0)
			throw "Failed to read " + path;

		if (targetDB[i] == nullptr) {
			targetDB[i] = make_shared<Shader>();
		}

		auto[ok, shader, error] = compileShader(src);

		if (ok)
			*targetDB[i] = shader;
		else {			
			errString.append(error);				
		}
	}

	return errString;
}
