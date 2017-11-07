#pragma once

#include "App.h"

#include "render/Camera.h"
#include "render/VertexBuffer.h"
#include "render/VolumeRaycaster.h"
#include "render/GLRenderer.h"

#include "utility/Options.h"

#include <batterylib/include/Volume.h>

#include <batterylib/include/Transform.h>
#include <batterylib/include/SimulatedAnnealing.h>

#include "Ui.h"

#include <memory>


struct Shader;

#define OPTIONS_FILENAME "options.json"

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
	virtual void callbackChar(GLFWwindow * w, unsigned int code) override;

	virtual void resetSA();

	OptionSet _options;

	Camera _camera;
	
	
	std::unordered_map<std::string, std::shared_ptr<Shader>> _shaders;
	
	vec3 _quadric;

	blib::Volume<unsigned char> _volume;

	std::unique_ptr<VolumeRaycaster> _volumeRaycaster;

	bool _autoUpdate;

	blib::SimulatedAnnealing<
		std::vector<blib::Transform>
	> _sa;
	

	RenderList _glRenderer;

	friend Ui;
	Ui _ui;	

private:
	/*
		Throws on first time
	*/
	void reloadShaders(bool firstTime);

	
	

};