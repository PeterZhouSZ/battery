#include "BatteryApp.h"
#include "GLFW/glfw3.h"

#include "utility/IOUtility.h"



#include "render/MeshObject.h"

#include "render/VolumeRaycaster.h"
#include "render/Shader.h"

#include <batterylib/include/VolumeIO.h>
#include <batterylib/include/RandomGenerator.h>
#include <batterylib/include/GeometryIO.h>


#include <glm/gtc/matrix_transform.hpp>




#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <array>
#include <numeric>






using namespace std;
using namespace blib;

RNGNormal normalDist(0, 1);
RNGUniformFloat uniformDist(0, 1);
RNGUniformInt uniformDistInt(0, INT_MAX);


#include <batterylib/include/OrientationHistogram.h>
#include "render/PrimitivesVBO.h"
#include "render/Shaders.h"
#include "batterylib/include/MGGPU.h"

#include "batterylib/include/Timer.h"

#include "batterylib/include/VolumeMeasures.h"
#include "batterylib/include/VolumeRasterization.h"
#include "batterylib/include/GeometryIO.h"
#include "batterylib/include/GeometryObject.h"
#include "batterylib/include/SAP.h"
#include "batterylib/include/Intersection.h"




BatteryApp::BatteryApp()
	: App("BatteryViz"),
	_camera(Camera::defaultCamera(_window.width, _window.height)),	
	_ui(*this),
	_currentRenderChannel(0)	
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
	
		
	
	_autoUpdate = false;

	_aabbVBO = getCubeVBO();
	
	

	/*bool res = loadFromFile(_options["Input"].get<std::string>("DefaultPath"));
	if (!res)
		std::cerr << "Failed to load default path" << std::endl;*/
	

	

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
	
	{
		static float iso = _options["Render"].get<float>("MarchingCubesIso");
		static int res = _options["Render"].get<int>("MarchingCubesRes");		
		static float smooth = _options["Render"].get<float>("MarchingCubesSmooth");

		float newIso = _options["Render"].get<float>("MarchingCubesIso");
		float newSmooth = _options["Render"].get<float>("MarchingCubesSmooth");
		int newRes = _options["Render"].get<int>("MarchingCubesRes");
		if (newRes < 8) newRes = 8;

		if (newIso != iso || newRes != res || newSmooth != smooth) {
			res = newRes;
			iso = newIso;
			smooth = newSmooth;
			runAreaDensity();
		}		


	}

	

	
	
	
	if (!_autoUpdate) return;





	return;
}

void BatteryApp::render(double dt)
{
	

	if (_window.width == 0 || _window.height == 0) return;

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	
	
	float sliceHeight = 0;
	if (_options["Render"].get<bool>("slices")) {
		sliceHeight = 1.0f / 3.0f;
	}
	

	
	if (_options["Render"]["SceneGeometry"].get<bool>("Enabled")) {

		RenderList rl;

		bool bboxes = _options["Render"]["SceneGeometry"].get<bool>("BoundingBoxes");
		

		
		for (auto & p : _sceneGeometry) {
			
			Transform volumeTransform;
			volumeTransform.scale = vec3(2);
			volumeTransform.translation = vec3(-1);

			auto templateGeometry = p->getTemplateGeometry();
			auto vboIt = _geometryVBOs.find(templateGeometry);
			if (vboIt != _geometryVBOs.end()) {

				auto & vbo = vboIt->second;

				auto t = p->getTransform();
				{
					mat4 M = volumeTransform.getAffine() * t.getAffine();
					mat4 NM = mat4(glm::transpose(glm::inverse(mat3(M))));

					ShaderOptions so = { { "M", M },{ "NM", NM },{ "PV", _camera.getPV() },{ "viewPos", _camera.getPosition() } };
					RenderList::RenderItem item = { vbo, so, GL_FILL };
					rl.add(_shaders[SHADER_PHONG], item);
				}
			}

			if (bboxes) {
				//auto t = p->getTransform();
				auto bb = p->bounds();

				Transform tbb;
				tbb.scale = vec3(0.5f) * bb.range();
				tbb.translation = bb.centroid();
				
				{
					mat4 M = volumeTransform.getAffine() * tbb.getAffine();
					mat4 NM = mat4(glm::transpose(glm::inverse(mat3(M))));

					ShaderOptions so = { 
						{ "M", M },
						{ "NM", NM },
						{ "PV", _camera.getPV() },
						{ "useUniformColor", true},
						{ "uniformColor", vec4(1.0f,0,0,1.0f) }
					};
					RenderList::RenderItem item = { _aabbVBO, so, GL_LINE };
					rl.add(_shaders[SHADER_FLAT], item);
				}
			
			}

		}

		

		rl.render();

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
			
			EigenTransform boxT;
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
	ivec4 viewport = {
		0, _window.height * sliceHeight, _window.width, _window.height - _window.height * sliceHeight
	};

	if (_options["Render"].get<bool>("volume") && _volume->numChannels() > 0) {
		
		//update current channel to render
		_currentRenderChannel = _options["Render"].get<int>("channel");
		_currentRenderChannel = std::clamp(_currentRenderChannel, uint(0), _volume->numChannels()-1);
		_options["Render"].set<int>("channel", _currentRenderChannel);
		_volumeRaycaster->setVolume(*_volume, _currentRenderChannel);
		

		if (_options["Render"].get<bool>("transferDefault")) {
			if (_volume->getChannel(_currentRenderChannel).type() == TYPE_UCHAR)
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

		_volumeRaycaster->render(_camera, viewport);	

	}

	if (_options["Render"].get<bool>("volumeGrid")) {
		_volumeRaycaster->renderGrid(_camera, viewport, *_shaders[SHADER_FLAT],
			_options["Render"].get<float>("volumeGridOpacity")

		);
	}

	//Render marching cubes volume
	{
		RenderList rl;

		mat4 M = mat4(1.0f);
		ShaderOptions so = {
			{ "M", M },
			{ "NM", M },
			{ "PV", _camera.getPV() }			
		};


		const GLenum fill = _options["Render"].get<bool>("MarchingCubesWire") ? GL_LINE : GL_FILL;

		RenderList::RenderItem item = { _volumeMC, so, fill, GL_BACK };
		rl.add(_shaders[SHADER_PHONG], item);
		rl.render();
	}

	
	/*
	Volume slices
	*/	
	if (_options["Render"].get<bool>("slices")) {
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
		UI render and update
	*/
	_ui.update(dt);
}



void BatteryApp::runAreaDensity()
{
	/*uint vboIndex;
	size_t Nverts = 0;

	float iso = _options["Render"].get<float>("MarchingCubesIso");
	int res = _options["Render"].get<int>("MarchingCubesRes");
	auto & mask = _volume->getChannel(CHANNEL_MASK);
	//float smooth = float(res) / mask.dim().x;
	
	//ivec3 mcres = mask.dim();
	ivec3 mcres = ivec3(res);

	double a = blib::getReactiveAreaDensity<double>(mask, mcres, iso, 1.0f, &vboIndex, &Nverts);

	int nparticles = _options["Generator"]["Spheres"].get<int>("N");
	double porosity = blib::getPorosity<double>(_volume->getChannel(CHANNEL_MASK));
	double volume = 1.0 - porosity;
	std::cout << "Porosity : " << porosity << ", Particle volume: " << volume << std::endl;
	double sf = blib::getShapeFactor(a / nparticles, volume / nparticles);

	size_t N = mask.dim().x * mask.dim().y  *mask.dim().z;

	std::cout << "Reactive Area Density: " << a << ", Shape Factor: " << sf << " per " << nparticles << " particles, normalized a: " << a / N << "\n";	
	
	if (Nverts > 0) {
		_volumeMC = std::move(VertexBuffer<VertexData>(vboIndex, Nverts));
	}
	else {
		_volumeMC = std::move(VertexBuffer<VertexData>());
	}*/
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
			std::cout << "Reloading shaders ...";
			std::cerr << loadShaders(_shaders) << std::endl;			
			std::cout << "Done." << std::endl;
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

bool BatteryApp::loadFromFile(const std::string & folder)
{

	try {
		VolumeChannel c = loadTiffFolder(folder.c_str());

		ivec3 userSize = ivec3(_options["Input"].get<int>("GenResolution"));
		ivec3 size = glm::min(c.dim(), userSize);
		c.resize(ivec3(0), ivec3(size));

		return loadFromMask(std::move(c));		
	}
	catch (const char * msg) {
		std::cerr << msg << std::endl;
		return false;
	}

	return true;
}


bool BatteryApp::loadFromPosFile(const std::string & path, ivec3 resolution)
{

	reset();
	
	_volume->addChannel(resolution, TYPE_UCHAR, false, "posFileRasterized");
	auto concetrationID = _volume->addChannel(
		_volume->getChannel(CHANNEL_MASK).dim(),
		TYPE_DOUBLE
	);
	_volume->getChannel(CHANNEL_CONCETRATION).clear();


	std::ifstream f(path);
	_sceneGeometry = blib::readPosFile(f);
	
	if (_sceneGeometry.size() == 0)
		return false;

	std::shared_ptr<blib::Geometry> templateParticle;
	std::vector<mat4> matrixTransforms;
	

	for (auto obj : _sceneGeometry) {
		matrixTransforms.push_back(obj->getTransform().getAffine());

		//Generate vbos of template particles
		auto vboIt = _geometryVBOs.find(obj->getTemplateGeometry());

		if (vboIt == _geometryVBOs.end()) {
			_geometryVBOs[obj->getTemplateGeometry()] = getTriangleMeshVBO(
				*static_pointer_cast<blib::TriangleMesh>(obj->getTemplateGeometry()),
				vec4(0.5f, 0.5f, 0.5f, 1.0f)
			);
			templateParticle = obj->getTemplateGeometry();
		}

	}
	f.close();


	blib::TriangleMesh & templateMesh = *static_pointer_cast<blib::TriangleMesh>(templateParticle);

	/*
		TODO heterogenous particles -> map templates to transforms
	*/
	rasterize(
		(float*)templateMesh.getTriangleArray().data(),
		templateMesh.faces.size(),
		(float*)matrixTransforms.data(),
		matrixTransforms.size(),
		_volume->getChannel(CHANNEL_MASK)
	);

	return true;
}

bool BatteryApp::loadFromMask(blib::VolumeChannel && mask)
{

	reset();

	_volume->emplaceChannel(std::move(mask));
	_volume->binarize(CHANNEL_MASK, 1.0f);

	auto concetrationID = _volume->addChannel(
		_volume->getChannel(CHANNEL_MASK).dim(),
		TYPE_DOUBLE
	);
	_volume->getChannel(CHANNEL_CONCETRATION).clear();


	_volume->getChannel(CHANNEL_MASK).setName("Mask");
	_volume->getChannel(CHANNEL_CONCETRATION).setName("Concetration");
	_volume->getChannel(CHANNEL_MASK).getCurrentPtr().createTexture();
	runAreaDensity();
	_volumeRaycaster->setVolume(*_volume, 0);

	return true;
}

void BatteryApp::reset()
{
	_volume = make_unique<blib::Volume>();
	_volumeMC = std::move(VertexBuffer<VertexData>());


	

	return;

	int genResX = _options["Generator"].get<int>("Resolution");
	ivec3 genRes = ivec3(genResX);

	_volume->addChannel(genRes, TYPE_UCHAR, false, "rasterized");




	//Setup template particle
	if(false)
	{
		const std::string path = "../../data/particles/C3.txt";
		std::shared_ptr<blib::Geometry> templateParticle = std::move(blib::loadParticleMesh(path).normalized(true));


		_geometryVBOs[templateParticle] = getTriangleMeshVBO(
			*static_pointer_cast<blib::TriangleMesh>(templateParticle),
			vec4(0.5f, 0.5f, 0.5f, 1.0f)
		);
	}




	std::vector<mat4> matrixTransforms;
	int N = _options["Generator"].get<int>("NumParticles");
	
	_sceneGeometry.clear();
	Timer tgen(true);

	/*for (auto i = 0; i < N; i++) {
		blib::Transform T0;
		
		T0.scale = vec3(uniformDist.next() * 0.5 + 0.5f);
		T0.scale *= 0.25f;
		T0.rotation = glm::rotate(glm::quat(), uniformDist.next() * 2.0f * glm::pi<float>(), { uniformDist.next(),uniformDist.next(), uniformDist.next() });
		T0.translation = vec3(uniformDist.next(), uniformDist.next(), uniformDist.next());

		matrixTransforms.push_back(T0.getAffine());	

		std::shared_ptr<blib::GeometryObject> obj = std::make_shared<blib::GeometryObject>(
			templateParticle
			);
		obj->setTransform(T0);
		_sceneGeometry.push_back(std::move(obj));


	}*/

	std::ifstream f("../../data/shapes/AcuteGoldenRhombohedron/log_shape1.pf49.s1.pos");
	_sceneGeometry = blib::readPosFile(f);

	if (_sceneGeometry.size() == 0)
		return;

	std::shared_ptr<blib::Geometry> templateParticle;

	for(auto obj : _sceneGeometry){
		matrixTransforms.push_back(obj->getTransform().getAffine());

		auto vboIt = _geometryVBOs.find(obj->getTemplateGeometry());
		if (vboIt == _geometryVBOs.end()) {
			_geometryVBOs[obj->getTemplateGeometry()] = getTriangleMeshVBO(
				*static_pointer_cast<blib::TriangleMesh>(obj->getTemplateGeometry()),
				vec4(0.5f, 0.5f, 0.5f, 1.0f)
			);
			templateParticle = obj->getTemplateGeometry();
		}
	}
	f.close();
	
	std::cout << "Generating " << tgen.timeMs() << "ms" << std::endl;


	/*{
		Timer tbuild(true);
		blib::SAP sap;
		sap.build(_sceneGeometry);
		std::cout << "Building structure " << tbuild.timeMs() << "ms" << std::endl;

		Timer tcollision(true);
		auto pairs = std::move(sap.getCollisionPairs());

		size_t intersectionCount = 0;
		for (auto p : pairs) {
			bool res = isectTest(*p.first->getGeometry(), *p.second->getGeometry());
			if (res) intersectionCount++;
		}

		tcollision.stop();
		std::cout << "Test pairs: " << pairs.size() << ", collisions: " << intersectionCount << ", time: " << tcollision.timeMs() << "ms" << std::endl;
	}*/

	blib::TriangleMesh & templateMesh = *static_pointer_cast<blib::TriangleMesh>(templateParticle);
	
	rasterize(
		(float*)templateMesh.getTriangleArray().data(),
		templateMesh.faces.size(),
		(float*)matrixTransforms.data(),
		matrixTransforms.size(),
		_volume->getChannel(CHANNEL_MASK)
	);
		

	


	
	


	
}
