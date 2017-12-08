#include "BatteryApp.h"
#include "GLFW/glfw3.h"

#include "utility/IOUtility.h"



#include "render/MeshObject.h"

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

#define DATA_FOLDER "../../data/graphite/SL43_C5_1c5bar_Data/"


using namespace std;
using namespace blib;

RNGNormal normalDist(0, 1);
RNGUniformFloat uniformDist(0, 1);
RNGUniformInt uniformDistInt(0, INT_MAX);


#include <batterylib/include/OrientationHistogram.h>
#include "render/PrimitivesVBO.h"
#include "render/Shaders.h"




BatteryApp::BatteryApp()
	: App("BatteryViz"),
	_camera(Camera::defaultCamera(_window.width, _window.height)),	
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
	
	{
		auto errMsg = loadShaders(_shaders);
		if (errMsg != "")
			throw errMsg;
	}

	_volumeRaycaster = make_unique<VolumeRaycaster>(
		_shaders[SHADER_POSITION],
		_shaders[SHADER_VOLUME_RAYCASTER],
		_shaders[SHADER_VOLUME_SLICE]
	);
	
		
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


	/*
		Scene init
	*/

	auto sphereObj = make_shared<MeshObject>(blib::generateSphere());
	_scene.addObject("sphere", sphereObj);

	
	resetSA();
	_volumeRaycaster->updateVolume(_volume);
	

}





void BatteryApp::update(double dt)
{
	if (!_autoUpdate) return;

	
	_saEllipsoid.update(_options["Optim"].get<int>("stepsPerFrame"));
	_volumeRaycaster->updateVolume(_volume);	

	return;
}

void BatteryApp::render(double dt)
{
	

	if (_window.width == 0 || _window.height == 0) return;

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	
	

	/*
		Volume slices
	*/
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





	/*
		Gl renderer
	*/
	if (_options["Render"].get<bool>("scene")) {
		auto renderList = getSceneRenderList(_scene, _shaders, _camera);
		renderList.render();	
	}

	if (_options["Render"].get<bool>("ellipsoids")) {

		RenderList rl;

		static auto vboSphere = getSphereVBO();	

		for (auto & e : _saEllipsoid.state) {
			auto T = e.transform.getAffine();
			{
				mat4 M = *reinterpret_cast<const mat4*>(T.data());
				mat4 NM = mat4(glm::transpose(glm::inverse(mat3(M))));

				ShaderOptions so = { { "M", M },{ "NM", NM },{ "PV", _camera.getPV() },{ "viewPos", _camera.getPosition() } };
				RenderList::RenderItem item = { vboSphere, so };
				rl.add(_shaders[SHADER_PHONG], item);
			}
		}

		rl.render();		
	}

	if (_options["Render"].get<bool>("ellipsoidsBounds")) {
		RenderList rl;

		static auto vboCube = getCubeVBO();

		for (auto & e : _saEllipsoid.state) {
			auto bounds = e.aabb();
			
			Transform boxT;
			//Cube VBO is from -1 to 1
			boxT.scale = (bounds.max - bounds.min);
			boxT.translation = (bounds.max + bounds.min) * 0.5f;


			mat4 M = *reinterpret_cast<const mat4*>(boxT.getAffine().data()) *
				glm::scale(mat4(), vec3(0.5f));
			ShaderOptions so = { { "M", M },{ "NM", M },{ "PV", _camera.getPV() },
				{ "useUniformColor", true },{ "uniformColor", vec4(1,0,0,1) }
			};
			RenderList::RenderItem item = { vboCube, so, GL_LINE };
			rl.add(_shaders[SHADER_FLAT], item);
			
		}

		rl.render();
	
	}

	/*
	Volume raycaster
	*/
	if (_options["Render"].get<bool>("volume")) {
		_camera.setWindowDimensions(_window.width, _window.height - static_cast<int>(_window.height * sliceHeight));
		_volumeRaycaster->render(_camera, {
			0, _window.height * sliceHeight, _window.width, _window.height - _window.height * sliceHeight
		}, *_shaders[SHADER_POSITION], *_shaders[SHADER_VOLUME_RAYCASTER]);
	}

	/*
		UI render and update
	*/
	_ui.update(dt);
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
		
		if (key == GLFW_KEY_R) {
			std::cerr << loadShaders(_shaders) << std::endl;			
		}

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



	_saEllipsoid.score = [&](const vector<Ellipsoid> & vals) {

		vector<Eigen::Affine3f> transforms(vals.size());
		for (auto i = 0; i < vals.size(); i++) {
			transforms[i] = vals[i].transform.getAffine().inverse();
		}

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

					for (auto i = 0; i < vals.size(); i++) {

						auto inSpace = transforms[i] * pos;
						
						if (vals[i].isPointIn(inSpace)) {
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
/*

		int collisions = 0;
		int pairs = 0;
		for (auto i = 0; i < vals.size(); i++) {
			auto & ei = vals[i];
			for (auto j = i; j < vals.size(); j++) {
				auto & ej = vals[i];
				pairs++;
				if (blib::ellipsoidEllipsoidMonteCarlo(ei, ej, uniformDist,128)) {
					collisions++;
				}
			}
		}*/

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
		//score += (collisions / static_cast<float>(pairs)) * 10;


		return score * 1000;		
	};

	_saEllipsoid.getNeighbour = [&](const vector<Ellipsoid> & vals) {

		const float step = _options["Optim"]["SA"].get<float>("neighbourStep");

		vector<Ellipsoid> newVals = vals;

		using namespace Eigen;

		auto &v = newVals[uniformDistInt.next() % newVals.size()].transform;
		
		if (_options["Optim"]["SA"].get<bool>("doScale")){
			v.scale += 0.5f * step * Vector3f(normalDist.next(), normalDist.next(), normalDist.next());
			v.scale = v.scale.cwiseMax(Vector3f{ 0.1f,0.1f,0.1f });
			v.scale = v.scale.cwiseMin(Vector3f{ 1.0f,1.0f,1.0f });
		}
		
		if (_options["Optim"]["SA"].get<bool>("doTranslate")) {
			v.translation += step * Vector3f(normalDist.next(), normalDist.next(), normalDist.next());
			v.translation = v.translation.cwiseMax(Vector3f{ -1,-1,-1 });
			v.translation = v.translation.cwiseMin(Vector3f{ 1, 1, 1 });
		}

		if (_options["Optim"]["SA"].get<bool>("doRotate")) {			
			Quaternionf q[3];
			float angleStep = step;
			q[0] = AngleAxisf(step * normalDist.next(), Vector3f{ 1,0,0 });
			q[1] = AngleAxisf(step * normalDist.next(), Vector3f{ 0,1,0 });
			q[2] = AngleAxisf(step * normalDist.next(), Vector3f{ 0,0,1 });
			v.rotation = v.rotation * q[0] * q[1] * q[2];			
		}


		return newVals;
	};

	vector<Ellipsoid> initVec(_options["Optim"]["SA"].get<int>("N"));

	
	const Eigen::Vector3f alignmentAxis = Eigen::Vector3f::UnitX();
	const float MRD = _options["Optim"]["SA"].get<float>("MRD");
	const float MRDDelta = _options["Optim"]["SA"].get<float>("MRDDeltaRad");
	const float scaleInit = _options["Optim"]["SA"].get<float>("scaleInit");
	const float scaleMultMin = _options["Optim"]["SA"].get<float>("scaleMultMin");
	const float scaleMultMax = _options["Optim"]["SA"].get<float>("scaleMultMax");	

	for (auto & v : initVec) {
						
		auto & T = v.transform;	

		T.scale = Eigen::Vector3f(scaleInit, scaleInit, scaleInit) + 
			Eigen::Vector3f(
				0, 1.0f + normalDist.next(), 0
			);
		
		T.scale = T.scale.cwiseMax(scaleInit * scaleMultMin);
		T.scale = T.scale.cwiseMin(scaleInit * scaleMultMax);

		T.translation = Eigen::Vector3f(normalDist.next(), normalDist.next(), normalDist.next()) * 1.0f;// - Eigen::Vector3f(1.0f,1.0f,1.0f);
		T.translation = T.translation.cwiseMax(Eigen::Vector3f{ -1,-1,-1 });
		T.translation = T.translation.cwiseMin(Eigen::Vector3f{ 1, 1, 1 });

		//;

		auto targetAxis = randomOrientationGauss(normalDist,uniformDist, MRD, alignmentAxis, MRDDelta, true);
		auto longestAxis = v.longestAxis().normalized();
		auto rotAxis = longestAxis.cross(targetAxis);
		float angle = acos(targetAxis.dot(longestAxis));
		T.rotation = Eigen::AngleAxisf(angle, rotAxis);		
	
	}

	_saEllipsoid.getTemperature = temperatureExp;

	_saEllipsoid.init(initVec, _options["Optim"].get<int>("maxSteps"));
	

}
