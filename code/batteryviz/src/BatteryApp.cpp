#include "BatteryApp.h"
#include "GLFW/glfw3.h"

#include "utility/IOUtility.h"


#include "render/VolumeRaycaster.h"
#include "render/Shader.h"

#include <batterylib/include/VolumeIO.h>
#include <batterylib/include/RandomGenerator.h>


#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <array>
#include <numeric>


//#define EIGEN_USE_THREADS
//#include <unsupported/Eigen/CXX11/ThreadPool>


#define DATA_FOLDER "../../data/graphite/SL43_C5_1c5bar_Data/"
#define SHADER_PATH "../batteryviz/src/shaders/"


using namespace std;
using namespace blib;

RNGNormal normalDist(0, 1);
RNGUniformFloat uniformDist(0, 1);
RNGUniformInt uniformDistInt(0, INT_MAX);


#include <batterylib/include/OrientationHistogram.h>

void quickTestFunc() {

	for (auto k = 0; k < 256; k++) {
		OrientHistogram oh(8, 8, 8);


		const float pi = glm::pi<float>();
		const vec3 minO = vec3(-pi);
		const vec3 maxO = vec3(pi);



		for (auto i = 0; i < 25000000; i++) {
			//vec3 dir = { normalDist.next() * pi, normalDist.next() * pi, normalDist.next() * pi };
			//dir *= 0.5f;
			//dir += 0.0f;
			vec3 dir = { uniformDist.next() * pi, uniformDist.next() * pi, uniformDist.next() * pi };
			dir = 2.0f * dir - vec3(pi);
			dir = glm::clamp(dir, minO, maxO);
			oh.add(dir.x, dir.y, dir.z);
		}

		OrientDistribution od(oh);

		std::cout << od.getMRD({ 0,0,0 }) << ", " << od.getMRD({ -3.14f,-3.14f,-3.14f }) << std::endl;
		


		char breakpt;
		breakpt = 0;

	}
}


static const std::vector<string> shaderNames = {
	"volumeslice",
	"volumeraycast",
	"position"
};

BatteryApp::BatteryApp()
	: App("BatteryViz"),
	_camera(Camera::defaultCamera(_window.width, _window.height)),	
	_ui(*this)
	
{	

	//quickTestFunc();


	
	{
		std::ifstream optFile(OPTIONS_FILENAME);
		if (optFile.good())
			optFile >> _options;
		else
			throw string("Options file not found");
	}

	resetGL();
	reloadShaders(true);

	_volumeRaycaster = make_unique<VolumeRaycaster>(
		_shaders["position"],
		_shaders["volumeraycast"],
		_shaders["volumeslice"]
	);


	_blackOpacity = 0.001f;
	_whiteOpacity = 0.05f;	
	_quadric = { 0.988,1.605,1.084 };
	_autoUpdate = false;

	/*
		Load data
	*/
	bool loadDefualt = false;
#ifdef _DEBUG
	loadDefualt = false;
#endif


	if(loadDefualt)
	{

		//propagate exception
		_volume = loadTiffFolder(DATA_FOLDER);				
		_volumeRaycaster->updateVolume(_volume);
		
	}
	else {
		const int res = 32;

		_volume = emptyVolume<unsigned char>(res);
		//_volume.resize(res,res,res);
		_volume.setZero();
		//_volume.resize({ res, res, res}, 0);		
		_autoUpdate = true;
	//	update(0);
		_autoUpdate = false;

			
	}

	
	resetSA();
	_volumeRaycaster->updateVolume(_volume);
	

}





void BatteryApp::update(double dt)
{
	if (!_autoUpdate) return;

	_sa.update(_options["Optim"].get<int>("stepsPerFrame"));
	_volumeRaycaster->updateVolume(_volume);

	std::cout << _sa.currentScore / 1000 << ", P:" << _sa.lastAcceptanceP << ", T: " << _sa.currentTemperature() << "\n";

	return;



	/*{
		auto & v = _volume;
		v = std::move(blib::diffuse(v, _options["Optim"].get<double>("diffusivity")));
		_volumeRaycaster->updateVolume(_volume);
		return;
	}*/




	
	
	//const int N = _options["Optim"].get<int>("N");
	const int N = 1;

		
	vector<mat4> transforms(N);
	
	float scaleMu = 0.25f;
	float scaleSigma = 0.15f;
	

	for (auto i = 0; i < N; i++) {
		float scale = scaleMu + normalDist.next() * scaleSigma;
		vec3 angles = { uniformDist.next(), uniformDist.next(), uniformDist.next() };
		vec3 offset = { uniformDist.next(), uniformDist.next(), uniformDist.next() };
		angles *= glm::pi<float>();		
		offset = (offset* 2.0f) - vec3(1.0f);

		auto M = glm::translate(mat4(), offset) *
			glm::rotate(mat4(), angles[0], vec3(1, 1, 0)) *
			glm::rotate(mat4(), angles[1], vec3(0, 1, 0)) *
			glm::rotate(mat4(), angles[2], vec3(0, 0, 1)) *
			glm::scale(mat4(), vec3(scale));

		transforms[i] = glm::inverse(M);
	}

	float static t = 0.0f;
	

	transforms[0] = glm::inverse(glm::rotate(mat4(), t, vec3(0, 1.0, 0)));

	t += float(dt);


	vec3 q = _quadric;

	mat4 M = glm::translate(mat4(), vec3(-0.5f,0.4f, 0.1f)) * glm::rotate(mat4(), 0.4f, vec3(1,0,0)) * glm::scale(mat4(), vec3(0.2f));
	mat4 invM = glm::inverse(M);

	std::vector<vec3> quadrics;

	
	const auto quadricImplicit = [](const vec3 & pos, const vec3 & q) {
		return (glm::pow(glm::abs(pos.x), q.x) + glm::pow(glm::abs(pos.y), q.y) + glm::pow(glm::abs(pos.z), q.z));
	};


	
	_volume.setZero();

	const auto dims = _volume.dimensions();

	#pragma omp parallel for schedule(dynamic)
	for (auto x = 0; x < dims[0]; x++) {
		for (auto y = 0; y < dims[1]; y++) {
			for (auto z = 0; z < dims[2]; z++) {
				vec3 pos = vec3(x / float(dims[0]), y / float(dims[1]), z / float(dims[2])) * 2.0f - vec3(1.0f);

				for (auto & M : transforms) {
					vec3 inSpace = vec3(M * vec4(pos, 1));
					float v = quadricImplicit(inSpace, q);

					if (v < 1.0f) {
						_volume(x, y, z) = 255;
						break;
					}
				}
			}
		}
	}
	
	//_volume.convolve()

	//Eigen::SimpleThreadPool tpi(8);
	//Eigen::ThreadPoolDevice device(&tpi,8);

	/*struct op : Eigen::TensorCustomUnaryOp< {

	};*/

	//_volume.device(device) = _volume.unaryExpr([])

	
	
	/*_volume = _volume.unaryExpr([](unsigned char c) { 
		return 255; 
	});*/


	_volumeRaycaster->updateVolume(_volume);	
}

void BatteryApp::render(double dt)
{

	

	if (_window.width == 0 || _window.height == 0) return;

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	
	float sliceHeight = 0;

	if (_options["Render"].get<bool>("slices")){
		sliceHeight = 1.0f / 3.0f;

		mat3 R[3] = {
			mat3(),
			mat3(glm::rotate(mat4(), glm::radians(90.0f), vec3(0, 0, 0))),
			mat3(glm::rotate(mat4(), glm::radians(90.0f), vec3(1, 0, 0)))
		};

		int yoffset = 0;
		int ysize = static_cast<int>(_window.height  * sliceHeight);

		_volumeRaycaster->renderSlice(0, ivec2(_window.width / 3 * 0, yoffset), ivec2(_window.width / 3, ysize));
		_volumeRaycaster->renderSlice(1, ivec2(_window.width / 3 * 1, yoffset), ivec2(_window.width / 3, ysize));
		_volumeRaycaster->renderSlice(2, ivec2(_window.width / 3 * 2, yoffset), ivec2(_window.width / 3, ysize));
	}

	_camera.setWindowDimensions(_window.width, _window.height  - static_cast<int>(_window.height * sliceHeight));
	_volumeRaycaster->render(_camera, {
		0, _window.height * sliceHeight, _window.width, _window.height - _window.height * sliceHeight
	}, *_shaders["position"], *_shaders["volumeraycast"]);



	/*
		UI render and update
	*/
	_ui.update(dt);


}



void BatteryApp::reloadShaders(bool firstTime)
{

	for (auto & name : shaderNames) {


		const auto path = SHADER_PATH + name + ".shader";
		auto src = readFileWithIncludes(path);

		if (src.length() == 0)
			throw "Failed to read " + path;

		if (_shaders.find(name) == _shaders.end()) {
			_shaders[name] = make_shared<Shader>();
		}

		auto[ok, shader, error] = compileShader(src);

		if (ok)
			*_shaders[name] = shader;
		else{		
			if (firstTime)
				throw error;
			else
				std::cerr << error << std::endl;
		}
			
	}

	std::cout << shaderNames.size() << " shaders " + string((firstTime) ? "" : "re")+ "loaded" << endl;
}





void BatteryApp::callbackMousePos(GLFWwindow * w, double x, double y)
{
	App::callbackMousePos(w, x, y);

	if (_input.mouseButtonPressed[GLFW_MOUSE_BUTTON_2]) {

		auto & cam = _camera;

		auto angle = (_input.mousePos - _input.mouseButtonPressPos[GLFW_MOUSE_BUTTON_2]) / 360.0f;
		std::swap(angle.x, angle.y);
		angle.y = -angle.y;

		auto sideAxis = glm::cross(cam.getUp(), cam.getDirection());

		auto cpos = glm::vec4(cam.getPosition() - cam.getLookat(), 1.0f);
		cpos = glm::rotate(glm::mat4(), angle.y, cam.getUp()) * glm::rotate(glm::mat4(), angle.x, sideAxis) * cpos;
		cpos += glm::vec4(cam.getLookat(), 0.0f);
		cam.setPosition(glm::vec3(cpos.x, cpos.y, cpos.z));

		_input.mouseButtonPressPos[GLFW_MOUSE_BUTTON_2] = _input.mousePos;	
	}

}

void BatteryApp::callbackMouseButton(GLFWwindow * w, int button, int action, int mods)
{
	
	App::callbackMouseButton(w, button, action, mods);
	_ui.callbackMouseButton(w, button, action, mods);
		
}

void BatteryApp::callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods)
{
	App::callbackKey(w, key, scancode, action, mods);
		

	if (action == GLFW_RELEASE) {
		
		if (key == GLFW_KEY_R)
			reloadShaders(false);

		if (key == GLFW_KEY_SPACE)
			_autoUpdate = !_autoUpdate;
	}


	_ui.callbackKey(w, key, scancode, action, mods);
}

void BatteryApp::callbackScroll(GLFWwindow * w, double xoffset, double yoffset)
{
	App::callbackScroll(w, xoffset, yoffset);
	
	auto & cam = _camera;
	auto delta = static_cast<float>(0.1 * yoffset);
	cam.setPosition(((1.0f - 1.0f*delta) * (cam.getPosition() - cam.getLookat())) + cam.getLookat());
	
	_ui.callbackScroll(w, xoffset, yoffset);
}

void BatteryApp::callbackChar(GLFWwindow * w, unsigned int code)
{
	App::callbackChar(w, code);
	_ui.callbackChar(w, code);
}

void BatteryApp::resetSA()
{
	/*
	Init SA
	*/

	_sa.score = [&](const vector<Transform> & vals) {


		vector<Eigen::Affine3f> transforms(vals.size());
		for (auto i = 0; i < vals.size(); i++) {
			transforms[i] = vals[i].getAffine().inverse();
		}

		const auto quadricImplicit = [](const Eigen::Vector3f & pos, const vec3 & q) {
			return (glm::pow(glm::abs(pos[0]), q.x) + glm::pow(glm::abs(pos[1]), q.y) + glm::pow(glm::abs(pos[2]), q.z));
		};


		_volume.setZero();
		const auto dims = _volume.dimensions();

		vector<int> collisions(dims[0]);

#pragma omp parallel for schedule(dynamic)
		for (auto x = 0; x < dims[0]; x++) {
			int thisColl = 0;
			for (auto y = 0; y < dims[1]; y++) {
				for (auto z = 0; z < dims[2]; z++) {
					Eigen::Vector3f pos =
						Eigen::Vector3f(x / float(dims[0]), y / float(dims[1]), z / float(dims[2])) * 2.0f
						- Eigen::Vector3f(1.0f, 1.0f, 1.0f);

					for (auto & M : transforms) {

						auto inSpace = M * pos;
						float v = quadricImplicit(inSpace, _quadric);

						if (v < 1.0f) {
							if (_volume(x, y, z) == 255)
								thisColl++;

							_volume(x, y, z) = 255;
						}
					}
				}
			}

			collisions[x] = thisColl;
		}

		int collTotal = std::accumulate(collisions.begin(), collisions.end(), 0);

		int sum = 0;
		//todo reduce
		for (auto x = 0; x < dims[0]; x++) {
			for (auto y = 0; y < dims[1]; y++) {
				for (auto z = 0; z < dims[2]; z++) {
					if (_volume(x, y, z) == 255)
						sum++;
				}
			}
		}

		//return 1.0f;
		auto totalVoxels = dims[0] * dims[1] * dims[2];

		float score = (1.0 - sum / float(totalVoxels));// +2 * collTotal / float(totalVoxels);

													   //std::cout << "porosity: " << score << "\n";

		score += 10 * collTotal / float(totalVoxels);


		return score * 1000;

		//Eigen::Tensor<unsigned char, 0> b = _volume.sum();

	};

	_sa.getNeighbour = [](const vector<Transform> & vals) {

		const float step = 0.01f * 15;

		vector<Transform> newVals = vals;

		using namespace Eigen;

		auto &v = newVals[uniformDistInt.next() % newVals.size()];

		//for (auto & v : newVals) {
		v.scale += 0.5f * step * Vector3f(normalDist.next(), normalDist.next(), normalDist.next());
		v.scale = v.scale.cwiseMax(Vector3f{ 0.1f,0.1f,0.1f });
		v.scale = v.scale.cwiseMin(Vector3f{ 1.0f,1.0f,1.0f });



		v.translation += step * Vector3f(normalDist.next(), normalDist.next(), normalDist.next());
		v.translation = v.translation.cwiseMax(Vector3f{ -1,-1,-1 });
		v.translation = v.translation.cwiseMin(Vector3f{ 1, 1, 1 });



		Quaternionf q[3];
		float angleStep = step;
		q[0] = AngleAxisf(step * normalDist.next(), Vector3f{ 1,0,0 });
		q[1] = AngleAxisf(step * normalDist.next(), Vector3f{ 0,1,0 });
		q[2] = AngleAxisf(step * normalDist.next(), Vector3f{ 0,0,1 });
		v.rotation = v.rotation * q[0] * q[1] * q[2];
		//}

		return newVals;
	};


	vector<Transform> initVec(_options["Optim"].get<int>("N"));

	for (auto & v : initVec) {
		v.scale = Eigen::Vector3f(normalDist.next(), normalDist.next(), normalDist.next()) * 0.4f + Eigen::Vector3f(0.2f, 0.2f, 0.2f);

		v.translation = Eigen::Vector3f(normalDist.next(), normalDist.next(), normalDist.next()) * 1.0f;// - Eigen::Vector3f(1.0f,1.0f,1.0f);
		v.translation = v.translation.cwiseMax(Eigen::Vector3f{ -1,-1,-1 });
		v.translation = v.translation.cwiseMin(Eigen::Vector3f{ 1, 1, 1 });
	}
	_sa.getTemperature = temperatureExp;

	_sa.init(initVec, _options["Optim"].get<int>("maxSteps"));


}
