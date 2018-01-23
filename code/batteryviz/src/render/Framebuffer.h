#pragma once
#include <batterylib/include/GLGlobal.h>

struct FrameBuffer {
	FrameBuffer();
	~FrameBuffer();
	GLuint ID() const;

private:
	GLuint _ID;
};