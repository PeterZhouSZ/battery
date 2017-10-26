#pragma once
#include "render/Texture.h"
#include "render/Camera.h"

#include "Volume.h"


struct FrameBuffer {
	FrameBuffer() {
		glGenFramebuffers(1, &_ID);
	}
	~FrameBuffer() {
		glDeleteFramebuffers(1, &_ID);
	}
	GLuint ID() const {
		return _ID;
	}

private:
	GLuint _ID;
};

struct EnterExitVolume {
	void resize(GLuint w, GLuint h);

	FrameBuffer enterFramebuffer;
	FrameBuffer exitFramebuffer;
	Texture enterTexture;
	Texture exitTexture;
};


struct VolumeRaycaster {

	bool updateVolume(const Volume<unsigned char> & volume);

	void render(const Camera & camera);

private:
	EnterExitVolume _enterExit;

	Texture _volumeTexture;
	Texture _transferTexture;

	vec3 _sliceMin;
	vec3 _sliceMax;

};