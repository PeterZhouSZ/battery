#pragma once

#include "App.h"

#include "render/Camera.h"
#include "render/Shader.h"
#include "render/Texture.h"
#include "render/VertexBuffer.h"

#include "Volume.h"

#include <memory>


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

class BatteryApp : public App {

public:
	BatteryApp();

	void renderSlice(Texture & texture, int axis, ivec2 screenPos, ivec2 screenSize, float t);
	
protected:
	virtual void update(double dt) override;
	virtual void render(double dt) override;

	virtual void callbackMousePos(GLFWwindow * w, double x, double y) override;
	virtual void callbackMouseButton(GLFWwindow * w, int button, int action, int mods) override;
	virtual void callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods) override;
	virtual void callbackScroll(GLFWwindow * w, double xoffset, double yoffset) override;


	Camera _camera;
	Texture _volumeTexture;
	Texture _transferTexture;
	

	VertexBuffer<VertexData> _quad;
	VertexBuffer<VertexData> _cube;
	EnterExitVolume _enterExit;

	std::unordered_map<std::string, std::shared_ptr<Shader>> _shaders;


	vec3 _sliceMin;
	vec3 _sliceMax;

	float _blackOpacity;
	float _whiteOpacity;

	vec3 _quadric;

	Volume<unsigned char> _volume;

private:
	void reloadShaders();

	
	void setVolumeTexture(Texture & tex, Volume<unsigned char> & volume);

};