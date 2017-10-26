#pragma once
#include "render/GLGlobal.h"

struct FrameBuffer {
	FrameBuffer();
	~FrameBuffer();
	GLuint ID() const;

private:
	GLuint _ID;
};