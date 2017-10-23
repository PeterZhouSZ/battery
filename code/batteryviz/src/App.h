#pragma once

struct GLFWwindow;

class App {
	
	struct Window {
		int width;
		int height;
		GLFWwindow * handle;
	};

	struct Input {
		bool mouseButtonPressed[8];

	};

public:
	App(const char * windowTitle);	
	bool run();
	void terminate();

	~App();

protected:	

	virtual void update(double dt);
	virtual void render(double dt);

	virtual void callbackMousePos(GLFWwindow * w, double x, double y);
	virtual void callbackMouseButton(GLFWwindow * w, int button, int action, int mods);
	virtual void callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods);
	virtual void callbackScroll(GLFWwindow * w, double xoffset, double yoffset);
	virtual void callbackResize(GLFWwindow * w, int width, int height);

	Window _window;
	Input _input;

	double _lastTime;	
private:	
	bool _terminate;

};