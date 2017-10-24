#include "BatteryApp.h"
#include "GLFW/glfw3.h"

#include "utility/IOUtility.h"

#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <iostream>
#include <stdexcept>
#include <filesystem>

#include "imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"

namespace fs = std::experimental::filesystem;
using namespace std;

#define DATA_FOLDER "../../../data/graphite/SL43_C5_1c5bar_Data/"


static const std::vector<string> shaderNames = {
	"volumeslice",
	"volumeraycast",
	"position"
};



/*
	Other functions, TODO: move to separate files
*/
VertexBuffer<VertexData> getQuadVBO() {

	VertexBuffer<VertexData> vbo;

	std::vector<VertexData> data = {
		VertexData({ -1.0,-1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0,-1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0,1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0,1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 })
	};
	vbo.setData(data.begin(), data.end());
	vbo.setPrimitiveType(GL_QUADS);

	return vbo;
}

VertexBuffer<VertexData> getCubeIVBO() {

	VertexBuffer<VertexData> ivbo;
		

	const std::vector<VertexData> data = {
		VertexData({ -1.0f, -1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, -1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, 1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0f, 1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0f, -1.0f, -1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, -1.0f, -1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, 1.0f, -1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0f, 1.0f, -1.0 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 })
	};


	const std::vector<unsigned int> indices = {
		0, 1, 2, 2, 3, 0,
		3, 2, 6, 6, 7, 3,
		7, 6, 5, 5, 4, 7,
		4, 0, 3, 3, 7, 4,
		0, 5, 1, 5, 0, 4,
		1, 5, 6, 6, 2, 1
	};		


	std::vector<VertexData> tridata;

	for (auto i = 0; i < indices.size(); i += 3) {
		auto a = indices[i];
		auto b = indices[i + 1];
		auto c = indices[i + 2];
		tridata.push_back(data[a]);
		tridata.push_back(data[b]);
		tridata.push_back(data[c]);
	}


	ivbo.setData(tridata.begin(), tridata.end());
	//ivbo.setIndices<unsigned int>(indices.begin(), indices.end(), GL_UNSIGNED_INT);
	ivbo.setPrimitiveType(GL_TRIANGLES);	

	return ivbo;
}


bool resetGL()
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_TEXTURE_3D);

	//GL(THIS_FUNCTION);
	return true;
}


void EnterExitVolume::resize(GLuint w, GLuint h)
{
	if (w != enterTexture.size.x || h != enterTexture.size.y) {
		GL(glBindFramebuffer(GL_FRAMEBUFFER, enterFramebuffer.ID()));
		GL(glBindTexture(GL_TEXTURE_2D, enterTexture.ID()));
		GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, NULL));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, enterTexture.ID(), 0));
		GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	}
	if (w != exitTexture.size.x || h != exitTexture.size.y) {
		GL(glBindFramebuffer(GL_FRAMEBUFFER, exitFramebuffer.ID()));
		GL(glBindTexture(GL_TEXTURE_2D, exitTexture.ID()));
		GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, NULL));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, exitTexture.ID(), 0));
		GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	}
}


BatteryApp::BatteryApp()
	: App("BatteryViz"),
	_camera(Camera::defaultCamera(_window.width, _window.height)),
	_quad(getQuadVBO()),
	_cube(getCubeIVBO())
{	
	resetGL();
	reloadShaders();

	_sliceMin = { -1,-1,-1 };
	_sliceMax = { 1,1,1 };
	_blackOpacity = 0.001;
	_whiteOpacity = 0.05;
	_quadric = { 0.617,0.617,0.482 };

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

		
		
		_volume.resize({ x,y,numSlices },0);
		auto & buffer = _volume.data;

		int cnt = 0;
		size_t bytesRead = 0;
		for (auto & f : fs::directory_iterator(path)) {
			size_t index = (cnt++) * (x * y * bytes);
			bytesRead += x * y * bytes;
			readTiff(f.path().string().c_str(), buffer.data() + index);				
		}		

		cout << "Read " << numSlices << " slices. (" << (bytesRead / (1024 * 1024)) << "MB)" << endl;

		
				
		
	}
	else {

		_volume.resize({ 128, 128, 128 }, 0);


			
	}

	_volumeTexture = Texture(GL_TEXTURE_3D, _volume.size.x, _volume.size.y, _volume.size.z);
	setVolumeTexture(_volumeTexture, _volume);


	{
		std::vector<color4> transferVal(16);
		for (auto & v : transferVal)
			v = color4(0.5f);

		transferVal[15] = vec4(0.0f);
		

		_transferTexture = Texture(GL_TEXTURE_1D, 16, 1, 0);
		glBindTexture(GL_TEXTURE_1D, _transferTexture.ID());
		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, transferVal.size(), 0, GL_RGBA, GL_FLOAT, transferVal.data());
		glBindTexture(GL_TEXTURE_1D, 0);
	}



	//gui
	ImGui_ImplGlfwGL3_Init(_window.handle, false);
	

}

void BatteryApp::renderSlice(Texture & texture, int axis, ivec2 screenPos, ivec2 screenSize, float t)
{
	GL(glDisable(GL_CULL_FACE));
	GL(glViewport(screenPos.x, screenPos.y, screenSize.x, screenSize.y));
	auto &shader = *_shaders["volumeslice"];

	shader.bind();
	
	//Set uniforms
	shader["tex"] = texture.bindTo(GL_TEXTURE0);
	shader["slice"] = (t + 1) / 2.0f;
	shader["axis"] = axis; 

	//Render fullscreen quad
	_quad.render();

	shader.unbind();
}

void BatteryApp::update(double dt)
{

	vec3 q = _quadric;

	_volume.clear(0);

	#pragma omp parallel for
	for (auto x = 0; x < _volume.size.x; x++) {
		for (auto y = 0; y < _volume.size.x; y++) {
			for (auto z = 0; z < _volume.size.z; z++) {
				vec3 pos = vec3(x / float(_volume.size.x), y / float(_volume.size.y), z / float(_volume.size.z)) * 2.0f - vec3(1.0f);

				float v = (glm::pow(glm::abs(pos.x), q.x) + glm::pow(glm::abs(pos.y), q.y) + glm::pow(glm::abs(pos.z), q.z));

				if (v < 1.0f) {
					_volume.at(x, y, z) = 255;
				}
			}
		}
	}

	setVolumeTexture(_volumeTexture, _volume);
}

void BatteryApp::render(double dt)
{

	if (_window.width == 0 || _window.height == 0) return;

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	
	{
		mat3 R[3] = {
			mat3(),
			mat3(glm::rotate(mat4(), glm::radians(90.0f), vec3(0, 0, 0))),
			mat3(glm::rotate(mat4(), glm::radians(90.0f), vec3(1, 0, 0)))
		};

		int yoffset = 0;
		int ysize = _window.height / 3;

		renderSlice(_volumeTexture, 0, ivec2(_window.width / 3 * 0, yoffset), ivec2(_window.width / 3, ysize),
			_sliceMin[0]);
		renderSlice(_volumeTexture, 1, ivec2(_window.width / 3 * 1, yoffset), ivec2(_window.width / 3, ysize),
			_sliceMin[1]);
		renderSlice(_volumeTexture, 2, ivec2(_window.width / 3 * 2, yoffset), ivec2(_window.width / 3, ysize),
			_sliceMin[2]);
	}


	glViewport(0, _window.height / 3, _window.width, (_window.height / 3) * 2);
	_camera.setWindowDimensions(_window.width, (_window.height / 3) * 2);

	_enterExit.resize(_window.width, _window.height);

	//Render enter/exit texture
	{

		auto & shader = *_shaders["position"];

		shader.bind();
		shader["PVM"] = _camera.getPV();
		shader["minCrop"] = _sliceMin;
		shader["maxCrop"] = _sliceMax;

		GL(glBindFramebuffer(GL_FRAMEBUFFER, _enterExit.enterFramebuffer.ID()));
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		glViewport(0,0, _window.width, _window.height);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_FRONT);
		_cube.render();

		GL(glBindFramebuffer(GL_FRAMEBUFFER, _enterExit.exitFramebuffer.ID()));
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		glViewport(0, 0, _window.width, _window.height);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		_cube.render();

		GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

		shader.unbind();

	}

	//Raycast
	{
		glViewport(0, _window.height / 3, _window.width, (_window.height / 3) * 2);
		glDisable(GL_CULL_FACE);

		auto & shader = *_shaders["volumeraycast"];

		shader.bind();
		shader["transferFunc"] = _transferTexture.bindTo(GL_TEXTURE0);
		shader["volumeTexture"] = _volumeTexture.bindTo(GL_TEXTURE1);
		shader["enterVolumeTex"] = _enterExit.enterTexture.bindTo(GL_TEXTURE2);
		shader["exitVolumeTex"] = _enterExit.exitTexture.bindTo(GL_TEXTURE3);

		shader["steps"] = 128;
		shader["transferOpacity"] = 0.5f;
		shader["blackOpacity"] = _blackOpacity;
		shader["whiteOpacity"] = _whiteOpacity;


		_quad.render();

		shader.unbind();
	
	}



	ImGui_ImplGlfwGL3_NewFrame();


	int w = _window.width * 0.2f;
	ImGui::SetNextWindowPos(ImVec2(_window.width - w, 0), ImGuiSetCond_Always);
	ImGui::SetNextWindowSize(ImVec2(w, 2.0f * (_window.height / 3.0f)), ImGuiSetCond_Always);
	
	static bool mainOpen = false;
	ImGui::Begin("Main", &mainOpen);

	ImGui::SliderFloat3("Slice (Min)", reinterpret_cast<float*>(&_sliceMin), -1, 1);
	ImGui::SliderFloat3("Slice (Max)", reinterpret_cast<float*>(&_sliceMax), -1, 1);

	ImGui::InputFloat("White opacity", &_whiteOpacity,0.001f);
	ImGui::InputFloat("Black opacity", &_blackOpacity, 0.001f);


	ImGui::SliderFloat3("Quadric", reinterpret_cast<float*>(&_quadric), 0, 10);

	ImGui::End();


	ImGui::Render();


}



void BatteryApp::reloadShaders()
{
	

	for (auto & name : shaderNames) {
		const auto path = "../src/shaders/" + name + ".shader";
		auto src = readFileWithIncludes(path);

		if (src.length() == 0)
			throw "Failed to read " + path;

		_shaders[name] = compileShader(src, [&](const string & msg) {
			std::cerr << toString("Failed to compile shader %, error:\n%", name, msg) << std::endl;
		});

		if (!_shaders[name])
			throw "Failed to compile " + name;
	}

	cout << shaderNames.size() << " shaders reloaded" << endl;

}



void BatteryApp::setVolumeTexture(Texture & tex, Volume<unsigned char> & volume)
{
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	GL(glBindTexture(GL_TEXTURE_3D, tex.ID()));
	GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, volume.size.x, volume.size.y, volume.size.z, 0, GL_RED, GL_UNSIGNED_BYTE, volume.data.data()));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
	GL(glBindTexture(GL_TEXTURE_3D, 0));
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
	ImGui_ImplGlfwGL3_MouseButtonCallback(w, button, action, mods);
}

void BatteryApp::callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods)
{
	App::callbackKey(w, key, scancode, action, mods);

	ImGui_ImplGlfwGL3_KeyCallback(w, key, scancode, action, mods);

	if (action == GLFW_RELEASE) {
		
		if (key == GLFW_KEY_R)
			reloadShaders();
	}

}

void BatteryApp::callbackScroll(GLFWwindow * w, double xoffset, double yoffset)
{
	App::callbackScroll(w, xoffset, yoffset);

	ImGui_ImplGlfwGL3_ScrollCallback(w, xoffset, yoffset);

	auto & cam = _camera;

	auto delta = static_cast<float>(0.1 * yoffset);
	cam.setPosition(((1.0f - 1.0f*delta) * (cam.getPosition() - cam.getLookat())) + cam.getLookat());

}