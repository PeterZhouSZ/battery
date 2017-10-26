#pragma once

#include "App.h"

#include "render/Camera.h"
#include "render/Shader.h"
#include "render/Texture.h"
#include "render/VertexBuffer.h"
#include "render/VolumeRaycaster.h"

#include "Volume.h"

#include <memory>





class BatteryApp : public App {

public:
	BatteryApp();
		
	
protected:
	virtual void update(double dt) override;
	virtual void render(double dt) override;

	virtual void callbackMousePos(GLFWwindow * w, double x, double y) override;
	virtual void callbackMouseButton(GLFWwindow * w, int button, int action, int mods) override;
	virtual void callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods) override;
	virtual void callbackScroll(GLFWwindow * w, double xoffset, double yoffset) override;


	Camera _camera;
	
	
	

	VertexBuffer<VertexData> _quad;
	
	std::unordered_map<std::string, std::shared_ptr<Shader>> _shaders;


	
	float _blackOpacity;
	float _whiteOpacity;

	vec3 _quadric;

	Volume<unsigned char> _volume;

	std::unique_ptr<VolumeRaycaster> _volumeRaycaster;

	bool _autoUpdate;

private:
	void reloadShaders();

	
	

};