#pragma once

#include "App.h"

#include "render/Camera.h"
#include "render/VertexBuffer.h"
#include "render/VolumeRaycaster.h"
#include "render/GLRenderer.h"

#include "utility/Options.h"

#include <batterylib/include/Volume.h>
#include <batterylib/include/Ellipsoid.h>

#include <batterylib/include/Transform.h>
#include <batterylib/include/SimulatedAnnealing.h>
#include <batterylib/include/DiffusionSolver.h>
#include <batterylib/include/MultigridSolver.h>

#include "Ui.h"

#include <memory>
#include "batterylib/include/MultigridGPU.h"
#include "batterylib/include/MGGPU.h"


struct Shader;

#define OPTIONS_FILENAME "options.json"
#define CHANNEL_BATTERY 0
#define CHANNEL_CONCETRATION 1

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

	void solveMultigridCPU();
	void solveMultigridGPU();

	virtual void reset();

	OptionSet _options;

	/*
		Render
	*/
	Camera _camera;	
	ShaderDB _shaders;

	uint _currentRenderChannel;
	std::unique_ptr<VolumeRaycaster> _volumeRaycaster;
	Scene _scene;
	
	
	vec3 _quadric;

	
	
	std::unique_ptr<blib::Volume> _volume;


	bool _autoUpdate;

	blib::SimulatedAnnealing<
		std::vector<blib::Transform>
	> _sa;


	blib::SimulatedAnnealing<
		std::vector<blib::Ellipsoid>
	> _saEllipsoid;


	blib::DiffusionSolver<double> _diffSolver;
	blib::MultigridSolver<double> _multiSolver;
	blib::MultigridGPU<double> _multigridGPUSolver;
	blib::MGGPU<double> _mggpu;


	friend Ui;
	Ui _ui;	

	
	double _simulationTime;
	double _convergenceTime;
	float _residual;
	

private:
	

	
	

};