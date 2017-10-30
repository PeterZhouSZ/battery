#include "BatteryApp.h"
#include "GLFW/glfw3.h"

#include "utility/IOUtility.h"
#include "utility/RandomGenerator.h"

#include "render/PrimitivesVBO.h"
#include "render/VolumeRaycaster.h"
#include "render/Shader.h"

#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <array>

//#include "imgui.h"
//#include "imgui/imgui_impl_glfw_gl3.h"

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/ThreadPool>

RNGNormal normalDist(0,1);
RNGUniformFloat uniformDist(0, 1);

namespace fs = std::experimental::filesystem;
using namespace std;

#define DATA_FOLDER "../../../data/graphite/SL43_C5_1c5bar_Data/"



static const std::vector<string> shaderNames = {
	"volumeslice",
	"volumeraycast",
	"position"
};

BatteryApp::BatteryApp()
	: App("BatteryViz"),
	_camera(Camera::defaultCamera(_window.width, _window.height)),
	_quad(getQuadVBO()),
	_ui(*this)
	
{	

	
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


	_blackOpacity = 0.001;
	_whiteOpacity = 0.05;
	//_quadric = { 0.617,0.617,0.482 };
	_quadric = { 0.988,1.605,1.084 };
	_autoUpdate = false;

	/*
		Load data
	*/
	if(false)
	{
		fs::path path(DATA_FOLDER);

		if (!fs::is_directory(path))
			throw "Volume directory not found";
			

		int numSlices = directoryFileCount(path.string().c_str());

		int x, y, bytes;
		if (!tiffSize(fs::directory_iterator(path)->path().string().c_str(), &x, &y, &bytes))
			throw "Couldn't read tiff file";

		if (bytes != 1)
			throw "only uint8 supported right now";

		
		_volume.resize(x, y, numSlices);		

		std::vector<unsigned char> buffer(x*y, 0);

		//auto & buffer = _volume.data();

		int cnt = 0;
		size_t bytesRead = 0;

		Eigen::Index sliceIndex = 0;
		for (auto & f : fs::directory_iterator(path)) {
			size_t index = (cnt++) * (x * y * bytes);
			bytesRead += x * y * bytes;
			//readTiff(f.path().string().c_str(), buffer.data() + index);				
			readTiff(f.path().string().c_str(), buffer.data());

			for (Eigen::Index i = 0; i < x; i++) {
				for (Eigen::Index j = 0; j < y; j++){
					_volume(i, j, sliceIndex) = buffer[i + j*x];
				}
			}

			sliceIndex = 0;
		}		

		

		cout << "Read " << numSlices << " slices. (" << (bytesRead / (1024 * 1024)) << "MB)" << endl;

		
		
		_volumeRaycaster->updateVolume(_volume);
		
	}
	else {
		const int res = 64;

		_volume = emptyVolume(64);
		_volume.resize(res,res,res);
		_volume.setZero();
		//_volume.resize({ res, res, res}, 0);		
		_autoUpdate = true;
		update(0);
		_autoUpdate = false;

			
	}

	
	

}



void BatteryApp::update(double dt)
{
	if (!_autoUpdate) return;

	
	const int N = _options["Optim"].get<int>("N");
		
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
		int ysize = _window.height  * sliceHeight;

		_volumeRaycaster->renderSlice(0, ivec2(_window.width / 3 * 0, yoffset), ivec2(_window.width / 3, ysize));
		_volumeRaycaster->renderSlice(1, ivec2(_window.width / 3 * 1, yoffset), ivec2(_window.width / 3, ysize));
		_volumeRaycaster->renderSlice(2, ivec2(_window.width / 3 * 2, yoffset), ivec2(_window.width / 3, ysize));
	}

	_camera.setWindowDimensions(_window.width, _window.height  - _window.height * sliceHeight);
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


		const auto path = "../src/shaders/" + name + ".shader";
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
