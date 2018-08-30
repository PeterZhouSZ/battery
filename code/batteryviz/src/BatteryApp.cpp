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


//#define DATA_FOLDER "../../data/graphite/SL43_C5_1c5bar_Data/"
#define DATA_FOLDER "../../data/graphiteSections/SL43_C5_1c5bar_Data/SL43_C5_1c5bar_section001/"
//#define DATA_FOLDER "../../data/graphiteSections/dataset0/SL44_C4_4bar_section003/"
//#define DATA_FOLDER "C:/!/battery/battery/code/python/vol"
//#define DATA_FOLDER "../../data/graphite/Cropped/"


using namespace std;
using namespace blib;

RNGNormal normalDist(0, 1);
RNGUniformFloat uniformDist(0, 1);
RNGUniformInt uniformDistInt(0, INT_MAX);


#include <batterylib/include/OrientationHistogram.h>
#include "render/PrimitivesVBO.h"
#include "render/Shaders.h"
#include "batterylib/include/MGGPU.h"





void generateSpheresVolume(blib::Volume & volume, uint sphereCount, float sphereRadius) {

	{
		auto & c = volume.getChannel(CHANNEL_BATTERY);
		
		uchar * arr = (uchar *)c.getCurrentPtr().getCPU();

		srand(0);


		std::vector<vec3> pos;
		std::vector<float> rad;

		for (auto i = 0; i < sphereCount; i++) {
			pos.push_back({ (rand() % 1024) / 1024.0f, (rand() % 1024) / 1024.0f, (rand() % 1024) / 1024.0f });

			rad.push_back({ (rand() % 1024) / 1024.0f * sphereRadius });

			//pos.push_back({ uniformDist.next(),uniformDist.next(),uniformDist.next() });
			//rad.push_back({ uniformDist.next()  * sphereRadius });
		}

		#pragma omp parallel for
		for (auto i = 0; i < c.dim().x; i++) {
			for (auto j = 0; j < c.dim().y; j++) {
				for (auto k = 0; k < c.dim().z; k++) {

					arr[i + j*c.dim().x + k*c.dim().y*c.dim().x] = 0;

					for (auto x = 0; x < pos.size(); x++) {
						if (glm::length(vec3(i / float(c.dim().x), j / float(c.dim().y), k / float(c.dim().z)) - pos[x]) < rad[x]) {
							arr[i + j*c.dim().x + k*c.dim().y*c.dim().x] = 255;
							break;
						}
					}


				}
			}
		}

		c.getCurrentPtr().commit();

	}

	volume.binarize(CHANNEL_BATTERY, 1.0f);
}


BatteryApp::BatteryApp()
	: App("BatteryViz"),
	_camera(Camera::defaultCamera(_window.width, _window.height)),	
	_ui(*this),
	_currentRenderChannel(0),
	_simulationTime(0.0f),
	_diffSolver(true),
	_multiSolver(true),
	_multigridGPUSolver(true)
{	


	
	blib::VolumeChannel::enableOpenGLInterop = true;

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

	
	//INIT
	reset();


	/*
		Scene init
	*/

/*
	auto sphereObj = make_shared<MeshObject>(blib::generateSphere());
	_scene.addObject("sphere", sphereObj);
*/

	


	
	

}





void BatteryApp::update(double dt)
{
	if (!_autoUpdate) return;
 
	
	//_saEllipsoid.update(_options["Optim"].get<int>("stepsPerFrame"));
	//_volumeRaycaster->updateVolume(_volume);	



	if (_volume->hasChannel(CHANNEL_CONCETRATION)) {

		

		for (auto i = 0; i < _options["Optim"].get<int>("stepsPerFrame"); i++) {
			/*_volume->heat(CHANNEL_CONCETRATION);			
			_volume->getChannel(CHANNEL_CONCETRATION).swapBuffers();*/

			float t0 = glfwGetTime();

			auto dir = Dir(_options["Diffusion"].get<int>("direction"));

			_volume->diffuse(
				CHANNEL_BATTERY,
				CHANNEL_CONCETRATION,				
				_options["Diffusion"].get<float>("voxelSize"),
				_options["Diffusion"].get<float>("D_zero"), //0.5f,
				_options["Diffusion"].get<float>("D_one"), //0.0001f
				_options["Diffusion"].get<float>("C_high"), 
				_options["Diffusion"].get<float>("C_low"), 
				dir
			);

			static int k = 0; k++;

			//if (k % 256 == 0) {
				auto dim = _volume->getChannel(CHANNEL_CONCETRATION).dim();
				_residual = _volume->getChannel(CHANNEL_CONCETRATION).differenceSum();// / (dim.x*dim.y*dim.z);				
				//_residual = 0.1;
				if (abs(_residual) < 0.000001f && _convergenceTime < 0.0f) {
					_convergenceTime = _simulationTime;
				}
			//}

			_volume->getChannel(CHANNEL_CONCETRATION).swapBuffers();

			float t1 = glfwGetTime();

			_simulationTime += t1 - t0;

			//_volume->getChannel(CHANNEL_CONCETRATION).getCurrentPtr().retrieve();

			//void * ptr = _volume->getChannel(CHANNEL_CONCETRATION).getCurrentPtr().getCPU();

			char b;
			b = 0;

		}

		

	}

	
	

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
		
		//update current channel to render
		_currentRenderChannel = _options["Render"].get<int>("channel");
		_currentRenderChannel = std::clamp(_currentRenderChannel, uint(0), _volume->numChannels()-1);
		_options["Render"].set<int>("channel", _currentRenderChannel);
		_volumeRaycaster->setVolume(*_volume, _currentRenderChannel);
		

		if (_options["Render"].get<bool>("transferDefault")) {
			if (_currentRenderChannel == CHANNEL_BATTERY)
				_volumeRaycaster->setTransferGray();
			else
				_volumeRaycaster->setTransferJet();
		}
		else {
			int t = _options["Render"].get<int>("transferFunc");
			if(t == 0)
				_volumeRaycaster->setTransferGray();
			else
				_volumeRaycaster->setTransferJet();
		}

		_volumeRaycaster->enableFiltering(
			_options["Render"].get<bool>("volumeFiltering")
		);

		
		
			

		_camera.setWindowDimensions(_window.width, _window.height - static_cast<int>(_window.height * sliceHeight));

		ivec4 viewport = {
			0, _window.height * sliceHeight, _window.width, _window.height - _window.height * sliceHeight
		};

		_volumeRaycaster->render(_camera, viewport);

		if (_options["Render"].get<bool>("volumeGrid")) {
			_volumeRaycaster->renderGrid(_camera, viewport, *_shaders[SHADER_FLAT],
				_options["Render"].get<float>("volumeGridOpacity")

			);
		}

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

	if (_ui.isFocused()) {
		_ui.callbackKey(w, key, scancode, action, mods);
		return;
	}


	App::callbackKey(w, key, scancode, action, mods);
		

	if (action == GLFW_RELEASE || action == GLFW_REPEAT) {
		
		if (key == GLFW_KEY_R) {
			std::cerr << loadShaders(_shaders) << std::endl;			
		}

		if (key == GLFW_KEY_SPACE)
			_autoUpdate = !_autoUpdate;

		if (key == GLFW_KEY_Q)
			reset();


		if (key == GLFW_KEY_1) {
			_options["Render"].get<int>("channel") = 0;
		}

		if (key == GLFW_KEY_2 && _volume->hasChannel(1)) {
			_options["Render"].get<int>("channel") = 1;
		}
		if (key == GLFW_KEY_3 && _volume->hasChannel(2)) {
			_options["Render"].get<int>("channel") = 2;
		}
		if (key == GLFW_KEY_4 && _volume->hasChannel(3)) {
			_options["Render"].get<int>("channel") = 3;
		}

		if (key == GLFW_KEY_RIGHT) {
			_options["Render"].get<int>("channel")++;
			
			if (_options["Render"].get<int>("channel") >= _volume->numChannels())
				_options["Render"].get<int>("channel") = 0;

		}

		if (key == GLFW_KEY_LEFT) {
			_options["Render"].get<int>("channel")--;
			if (_options["Render"].get<int>("channel") < 0)
				_options["Render"].get<int>("channel") = int(_volume->numChannels()) -1;
		}

	}

	
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



void BatteryApp::solveMultigridCPU()
{

	//		_volume->getChannel(CHANNEL_BATTERY).clear();

	auto & c = _volume->getChannel(CHANNEL_BATTERY);
	uchar* data = (uchar*)c.getCurrentPtr().getCPU();

	auto maxDim = std::max(c.dim().x, std::max(c.dim().y, c.dim().z));
	auto minDim = std::min(c.dim().x, std::min(c.dim().y, c.dim().z));
	auto exactSolveDim = 4;
	int levels = std::log2(minDim) - std::log2(exactSolveDim) + 1;
	//int levels = 2;


	Dir dir = Dir(_options["Diffusion"].get<int>("direction"));

	std::cout << "Multigrid solver levels " << levels << std::endl;

	_multiSolver.setVerbose(false);

	auto t0 = std::chrono::system_clock::now();
	vec3 cellDim = vec3(1.0 / maxDim);
	_multiSolver.prepare(*_volume,
		data,
		c.dim(),
		dir,
		_options["Diffusion"].get<float>("D_zero"),
		_options["Diffusion"].get<float>("D_one"),
		levels,
		cellDim
	);
	auto t1 = std::chrono::system_clock::now();

	std::chrono::duration<double> prepTime = t1 - t0;
	std::cout << "Prep time: " << prepTime.count() << std::endl;

	double tol = pow(10.0, -_options["Diffusion"].get<int>("Tolerance"));

	_multiSolver.solve(*_volume, tol, 1024);
	//_multiSolver.solve(*_volume, 1e-6, 3);



	//	_multiSolver.generateErrorVolume(*_volume);
	_multiSolver.setVerbose(true);
	_multiSolver.tortuosity(_volume->getChannel(CHANNEL_BATTERY), dir);
	auto t2 = std::chrono::system_clock::now();

	_multiSolver.resultToVolume(_volume->getChannel(CHANNEL_CONCETRATION));
	_volume->getChannel(CHANNEL_CONCETRATION).getCurrentPtr().commit();



	std::chrono::duration<double> solveTime = t2 - t1;


	std::cout << "Solve time: " << solveTime.count() << std::endl;
	std::cout << "Iterations: " << _multiSolver.iterations() << std::endl;
	std::cout << "--------------" << std::endl;

}

void BatteryApp::solveMultigridGPU()
{
	auto & c = _volume->getChannel(CHANNEL_BATTERY);
	auto maxDim = std::max(c.dim().x, std::max(c.dim().y, c.dim().z));
	auto minDim = std::min(c.dim().x, std::min(c.dim().y, c.dim().z));
	auto exactSolveDim = 4;
	int levels = std::log2(minDim) - std::log2(exactSolveDim) + 1;

	Dir dir = Dir(_options["Diffusion"].get<int>("direction"));

	auto t0 = std::chrono::system_clock::now();
	vec3 cellDim = vec3(1.0 / maxDim);


	//_multigridGPUSolver.setDebugVolume(_volume.get());


	_multigridGPUSolver.prepare(
		_volume->getChannel(CHANNEL_BATTERY),
		_volume->getChannel(CHANNEL_CONCETRATION),
		dir,
		_options["Diffusion"].get<float>("D_zero"),
		_options["Diffusion"].get<float>("D_one"),
		levels,
		cellDim
	);

//	return;


	auto t1 = std::chrono::system_clock::now();
	std::chrono::duration<double> prepTime = t1 - t0;
	std::cout << "GPU Prep time: " << prepTime.count() << std::endl;

	t1 = std::chrono::system_clock::now();
	_multigridGPUSolver.solve(
		1e-6,
		4096,
		MultigridGPU<double>::CycleType::W_CYCLE
	);
	auto t2 = std::chrono::system_clock::now();


	std::chrono::duration<double> solveTime = t2 - t1;
	std::cout << "GPU Solve time: " << solveTime.count() <<
		", iterations: " << _multigridGPUSolver.iterations() << std::endl;
}

void BatteryApp::reset()
{

	

	_simulationTime = 0.0f;
	_convergenceTime = -1.0f;

	bool loadDefault = _options["Input"].get<bool>("Default");

	
	_volume = make_unique<blib::Volume>();

	//loadDefault = false;

	if (loadDefault) {
		auto batteryID = _volume->emplaceChannel(loadTiffFolder(DATA_FOLDER));
		assert(batteryID == CHANNEL_BATTERY);


		//_volume->getChannel(CHANNEL_BATTERY).resize(ivec3(0), 2*ivec3(32,32,16)  );
		ivec3 userSize = ivec3(_options["Input"].get<int>("GenResolution"));
		ivec3 size = glm::min(_volume->getChannel(CHANNEL_BATTERY).dim(), userSize);

		_volume->getChannel(CHANNEL_BATTERY).resize(ivec3(0), ivec3(size)  );
		_volume->binarize(CHANNEL_BATTERY, 1.0f);

		//Add concetration channel
		auto concetrationID = _volume->addChannel(
			_volume->getChannel(CHANNEL_BATTERY).dim(),
			TYPE_FLOAT
		);
		assert(concetrationID == CHANNEL_CONCETRATION);

		_volume->getChannel(CHANNEL_CONCETRATION).clear();
				

		auto dim = _volume->getChannel(CHANNEL_BATTERY).dim();
		std::cout << "Resolution: " << dim.x << " x " << dim.y << " x " << dim.z <<
			" = " << dim.x*dim.y*dim.z << " voxels (" << (dim.x*dim.y*dim.z) / (1024 * 1024.0f) << "M)" << std::endl;


		if (_options["Input"].get<bool>("Sphere")) {
			auto d = _volume->getChannel(CHANNEL_BATTERY).dim();

			auto & c = _volume->getChannel(batteryID);
			uchar* data = (uchar*)c.getCurrentPtr().getCPU();

			for (auto i = 0; i < d[0] - 0; i++) {
				for (auto j = 0; j < d[1] - 0; j++) {
					for (auto k = 0; k < d[2] - 0; k++) {

						auto index = linearIndex(c.dim(), { i,j,k });



						vec3 normPos = { i / float(d[0] - 1),j / float(d[1] - 1), k / float(d[2] - 1), };


						int border = 0;
						if (i <= border || i >= d[0] - border - 1) {
							data[index] = 0;
						}
						/*else if (j <= border || j >= d[1] - border - 1) {
							data[index] = 0;
						}
						else if (k <= border|| k >= d[2] - border - 1) {
							data[index] = 0;
						}*/
						else {
							data[index] = 255;
						}


					/*	if (normPos.x < 0.1f  || normPos.x > 0.9f)
							data[index] = 0;						
						if (normPos.y < 0.1f || normPos.y > 0.9f)
							data[index] = 0;
						if (normPos.z < 0.1f || normPos.z > 0.9f)
							data[index] = 0;*/

					}
				}
			}

			c.getCurrentPtr().commit();
		}

	}
	else {
		int res = _options["Input"].get<int>("GenResolution");
		if (res == 0) {
			res = 32;
			_options["Input"].get<int>("GenResolution") = res;
		}				
		ivec3 d = ivec3(res, res, res);

		std::cout << "Resolution: " << d.x << " x " << d.y << " x " << d.z <<
			" = " << d.x*d.y*d.z << " voxels (" << (d.x*d.y*d.z) / (1024 * 1024.0f) << "M)" << std::endl;


		//ivec3 d = ivec3(16, 16, 15);
		//ivec3 d = ivec3(179, 163, 157);
		auto batteryID = _volume->addChannel(d, TYPE_UCHAR);

		//Add concetration channel
		auto concetrationID = _volume->addChannel(
			_volume->getChannel(CHANNEL_BATTERY).dim(),
			TYPE_FLOAT
		);
		assert(concetrationID == CHANNEL_CONCETRATION);

		bool genSphere = _options["Input"].get<bool>("Sphere");
		if (genSphere) {
		
			auto & c = _volume->getChannel(batteryID);
			uchar* data = (uchar*)c.getCurrentPtr().getCPU();

			for (auto i = 0; i < d[0] - 0; i++) {
				for (auto j = 0; j < d[1] - 0; j++) {
					for (auto k = 0; k < d[2] - 0; k++) {

						auto index = linearIndex(c.dim(), { i,j,k });



						if (i > d[0] / 2)
							data[index] = 255;
						else
							data[index] = 0;

						vec3 normPos = { i / float(d[0] - 1),j / float(d[1] - 1), k / float(d[2] - 1), };

						//if()

						//data[index] = normPos.x + 0.;

						/*/ *if (abs(normPos.x - 0.5f) < 0.25f)
							data[index] = 255;
						else
							data[index] = 0;
						* /*/

						float distC = glm::length(normPos - vec3(0.5f));
						//auto diff = glm::abs(normPos - vec3(0.5f));
						//float distC = glm::max(glm::max(diff.x, diff.y),-1.0f);
						if (distC < 0.45f && distC > 0.35f)
							data[index] = 255;
						else
							data[index] = 0;
					}
				}
			}

			//generateSpheresVolume(*_volume, 128, 0.15f);

			c.getCurrentPtr().commit();

			
		}
		
		//Test for fipy cmp
		/*auto & c = _volume->getChannel(batteryID);
		uchar* data = (uchar*)c.getCurrentPtr().getCPU();

		/ *
		x here is z in python
		solving for -x here  (dir=1)
		* /

		/ *for (auto i = 0; i <d[0] - 0; i++) {
			for (auto j = 0; j <d[1] - 1; j++) {
				for (auto k = 0; k < d[2] - 0; k++) {

					data[i + j*c.dim().x + k*c.dim().y*c.dim().x] = 255;
				}
			}
		}* /
		c.getCurrentPtr().commit();*/

		

	}
	
	_volume->getChannel(CHANNEL_BATTERY).setName("Battery");
	_volume->getChannel(CHANNEL_CONCETRATION).setName("Concetration");

	
	if(true){
		MGGPU<double>::Params p;
		p.levels = 5;
		p.dir = X_NEG;
		p.d0 = _options["Diffusion"].get<float>("D_zero");
		p.d1 = _options["Diffusion"].get<float>("D_one");
		
		auto & c = _volume->getChannel(CHANNEL_BATTERY);
		auto maxDim = std::max(c.dim().x, std::max(c.dim().y, c.dim().z));
		auto minDim = std::min(c.dim().x, std::min(c.dim().y, c.dim().z));
		auto exactSolveDim = 4;		
		p.cellDim = vec3(1.0 / maxDim);
		p.levels = std::log2(minDim) - std::log2(exactSolveDim) + 1;
		p.dir = Dir(_options["Diffusion"].get<int>("direction"));

		std::cout << "Multigrid solver levels " << p.levels << std::endl;		

		auto t0 = std::chrono::system_clock::now();		
		_mggpu.prepare(_volume->getChannel(CHANNEL_BATTERY), p, *_volume);
		auto t1 = std::chrono::system_clock::now();

		std::chrono::duration<double> prepTime = t1 - t0;
		std::cout << "Prep time: " << prepTime.count() << "s" << std::endl;


		/////
		//exit(0);
		/////

	}


	const bool multigridGPU = false;
	const bool multigridTest = false;


	if (multigridGPU){		
		solveMultigridGPU();	
	}
	
	if(multigridTest){
		solveMultigridCPU();
	}

	_volumeRaycaster->setVolume(*_volume, 0);
}
