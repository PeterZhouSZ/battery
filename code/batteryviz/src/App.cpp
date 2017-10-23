#include "App.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>


static App * currentApp = nullptr;

App::App(const char * windowTitle)
	: 
	_terminate(false),
	_lastTime(0.0)
{

	if (currentApp)
		throw "App is already running";
	currentApp = this;

	/*
		GLFW init
	*/
	glfwSetErrorCallback([](int error, const char * desc){
		throw desc;
	});
	glfwInit();				

	//Window setup	
	auto & wh = _window.handle;
	wh = glfwCreateWindow(2275, 1280, windowTitle, NULL, NULL);
	glfwSetWindowPos(wh, 0, 110);
	glfwMakeContextCurrent(wh);
	glfwGetFramebufferSize(wh, &_window.width, &_window.height);

	//GLEW init
	auto glewCode = glewInit();
	if (glewCode != GLEW_OK)
		throw glewGetErrorString(glewCode);
		
	//Callback setup
	glfwSetCursorPosCallback(wh, 
		[](GLFWwindow * w, double x, double y) { 
			currentApp->callbackMousePos(w, x, y); 
	});


	for (auto i = 0; i < 8; i++)
		_input.mouseButtonPressed[i] = false;
	glfwSetMouseButtonCallback(wh, 
		[](GLFWwindow * w, int button, int action, int mods) { 
		currentApp->callbackMouseButton(w, button, action, mods); 
	});
	
	
	glfwSetKeyCallback(wh,
		[](GLFWwindow * w, int key, int scancode, int action, int mods) {
		currentApp->callbackKey(w, key, scancode, action, mods);
	});

	glfwSetScrollCallback(wh,
		[](GLFWwindow * w, double xoffset, double yoffset) {
		currentApp->callbackScroll(w, xoffset, yoffset);
	});

	glfwSetWindowSizeCallback(wh,
		[](GLFWwindow * w, int width, int heigth) {
		currentApp->callbackResize(w, width, heigth);
	});

}

bool App::run()
{
	if (glfwWindowShouldClose(_window.handle) || _terminate) 
		return false;			

	glfwPollEvents();

	const auto currentTime = glfwGetTime();
	const auto dt = currentTime - _lastTime;

	//Update & render
	update(dt);
	render(dt);

	glfwSwapBuffers(_window.handle);
	_lastTime = currentTime;
	return true;
}

void App::terminate()
{
	_terminate = true;
}

App::~App()
{
	glfwTerminate();
	currentApp = false;
}

void App::update(double dt)
{
	//do nothing
}

void App::render(double dt)
{
	//do nothing
}

void App::callbackMousePos(GLFWwindow * w, double x, double y)
{

}

void App::callbackMouseButton(GLFWwindow * w, int button, int action, int mods)
{
	if (action == GLFW_PRESS) {
		_input.mouseButtonPressed[button] = true;
	}
	else if (action == GLFW_RELEASE) {
		_input.mouseButtonPressed[button] = false;
	}
}

void App::callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE)
		terminate();
}

void App::callbackScroll(GLFWwindow * w, double xoffset, double yoffset)
{

}

void App::callbackResize(GLFWwindow * w, int width, int height)
{
	_window.width = width;
	_window.height = height;
}

