#include "BatteryApp.h"


#include "utility/IOUtility.h"

#include <string>
#include <iostream>
#include <stdexcept>

using namespace std;


BatteryApp::BatteryApp()
	: App("BatteryViz")	
{
	
	{
		_volumeTexture = Texture(GL_TEXTURE_3D, 128, 128, 128);
	}


	{
		_camera = Camera::defaultCamera(_window.width, _window.height);
	}
	

	{
		std::vector<VertexData> data = {
			VertexData({ -1.0,-1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
			VertexData({ 1.0,-1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
			VertexData({ 1.0,1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
			VertexData({ -1.0,1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 })
		};
		_quad.setData(data.begin(), data.end());
		_quad.setPrimitiveType(GL_QUADS);
	}

	
	{
		const auto path = "../src/shaders/volume.shader";
		auto src = readFileWithIncludes(path);

		if (src.length() == 0)
			throw "Failed to read shader file";

		_volumeShader = compileShader(src, [](const string & msg) {
			std::cerr << toString("Failed to compile shader %, error:\n%", "volume", msg) << std::endl;
		});
	}

}

void BatteryApp::update(double dt)
{

}

void BatteryApp::render(double dt)
{
	_volumeShader->bind();

	_quad.render();

	_volumeShader->unbind();
}

void BatteryApp::callbackMousePos(GLFWwindow * w, double x, double y)
{
	
}

void BatteryApp::callbackMouseButton(GLFWwindow * w, int button, int action, int mods)
{
	
}

void BatteryApp::callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods)
{
	
}

void BatteryApp::callbackScroll(GLFWwindow * w, double xoffset, double yoffset)
{
	
}

