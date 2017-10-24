#include "GLError.h"
#include <GL/glew.h>

#include <iostream>




void logCerr(const char * label, const char * errtype){

	std::cerr << label << ": " << errtype << '\n';

}
#ifdef DEBUG
bool GLError(const char *label /*= ""*/,
	const std::function<void(const char *label, const char *errtype)>
	&callback)
{
	bool hasErr = false;
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR)
	{	
		if (!callback) return false;
		switch (err) {
		case GL_INVALID_ENUM: callback(label,"GL_INVALID_ENUM"); break;
		case GL_INVALID_VALUE: callback(label, "GL_INVALID_VALUE"); break;
		case GL_INVALID_OPERATION: callback(label, "GL_INVALID_OPERATION"); break;
		case GL_STACK_OVERFLOW: callback(label, "GL_STACK_OVERFLOW"); break;
		case GL_STACK_UNDERFLOW: callback(label, "GL_STACK_UNDERFLOW"); break;
		case GL_OUT_OF_MEMORY: callback(label, "GL_OUT_OF_MEMORY"); break;
		case GL_INVALID_FRAMEBUFFER_OPERATION: callback(label, "GL_INVALID_FRAMEBUFFER_OPERATION"); break;
		default: callback(label, "Unknown Error"); break;
		}
		hasErr = true;
	}

	return hasErr;
}
#endif

