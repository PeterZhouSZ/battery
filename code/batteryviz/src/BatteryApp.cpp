#include "BatteryApp.h"


#include "utility/IOUtility.h"

#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <iostream>
#include <stdexcept>


/*
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"*/
#include "tinytiff/tinytiffreader.h"

#include <filesystem>
namespace fs = std::experimental::filesystem;


using namespace std;


BatteryApp::BatteryApp()
	: App("BatteryViz"),
	_camera(Camera::defaultCamera(_window.width, _window.height))
{	
	
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
		const auto path = "../src/shaders/volumeslice.shader";
		auto src = readFileWithIncludes(path);

		if (src.length() == 0)
			throw "Failed to read shader file";

		_volumeSliceShader = compileShader(src, [](const string & msg) {
			std::cerr << toString("Failed to compile shader %, error:\n%", "volume", msg) << std::endl;
		});

		if (!_volumeSliceShader)
			throw "Failed";

	}


	{
		fs::path path("../../../data/graphite/SL43_C5_1c5bar_Data/");

		if (!fs::is_directory(path))
			throw "Volume directory not found";
		
		

		int numSlices = directoryFileCount(path.string().c_str());

		int x, y, bytes;
		if (!tiffSize(fs::directory_iterator(path)->path().string().c_str(), &x, &y, &bytes))
			throw "Couldn't read tiff file";

		if (bytes != 1)
			throw "only uint8 supported right now";



		std::vector<unsigned char> buffer(numSlices*x*y*bytes);	

		int cnt = 0;
		size_t bytesRead = 0;
		for (auto & f : fs::directory_iterator(path)) {
			size_t index = (cnt++) * (x * y * bytes);
			bytesRead += x * y * bytes;
			readTiff(f.path().string().c_str(), buffer.data() + index);				
		}		

		cout << "Read " << numSlices << " slices. (" << (bytesRead / (1024 * 1024)) << "MB)" << endl;


		glEnable(GL_TEXTURE_3D);
		_volumeTexture = Texture(GL_TEXTURE_3D, x, y, numSlices);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		GL(glBindTexture(GL_TEXTURE_3D, _volumeTexture.ID()));
		GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, x, y, numSlices, 0, GL_RED, GL_UNSIGNED_BYTE, buffer.data()));		
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		GL(glBindTexture(GL_TEXTURE_3D, 0));
	}


	

}

void BatteryApp::renderSlice(Texture & texture, mat3 transform, ivec2 screenPos, ivec2 screenSize)
{

	

	glViewport(screenPos.x, screenPos.y,screenSize.x, screenSize.y);
	

	auto &shader = *_volumeSliceShader;

	shader.bind();
	shader["tex"] = texture.bindTo(GL_TEXTURE0);
	shader["slice"] = float(_lastTime) / 8.0f;
	shader["R"] = transform; //mat3(glm::rotate(mat4(), glm::radians(45.0f), vec3(0, 1, 0)));
	_quad.render();
	shader.unbind();

}

void BatteryApp::update(double dt)
{
	
}

void BatteryApp::render(double dt)
{	

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	_camera.setWindowDimensions(_window.width, _window.height);

	{
		mat3 R[3] = {
			mat3(),
			mat3(glm::rotate(mat4(), glm::radians(90.0f), vec3(0, 1, 0))),
			mat3(glm::rotate(mat4(), glm::radians(90.0f), vec3(1, 0, 0)))
		};

		int yoffset = 0;
		int ysize = _window.height / 3;

		renderSlice(_volumeTexture, R[0], ivec2(_window.width / 3 * 0, yoffset), ivec2(_window.width / 3, ysize));
		renderSlice(_volumeTexture, R[1], ivec2(_window.width / 3 * 1, yoffset), ivec2(_window.width / 3, ysize));
		renderSlice(_volumeTexture, R[2], ivec2(_window.width / 3 * 2, yoffset), ivec2(_window.width / 3, ysize));
	}



	
}

void BatteryApp::callbackMousePos(GLFWwindow * w, double x, double y)
{
	
}

void BatteryApp::callbackMouseButton(GLFWwindow * w, int button, int action, int mods)
{
	
}

void BatteryApp::callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods)
{
	App::callbackKey(w, key, scancode, action, mods);
}

void BatteryApp::callbackScroll(GLFWwindow * w, double xoffset, double yoffset)
{
	
}

