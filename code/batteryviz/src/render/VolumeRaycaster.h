#pragma once
#include "render/Texture.h"
#include "render/Camera.h"
#include "render/Shader.h"
#include "render/VertexBuffer.h"
#include "render/Framebuffer.h"

#include "batterylib/include/Volume.h"



struct VolumeRaycaster {

	struct EnterExitVolume {
		EnterExitVolume();
		void resize(GLuint w, GLuint h);

		FrameBuffer enterFramebuffer;
		FrameBuffer exitFramebuffer;
		Texture enterTexture;
		Texture exitTexture;
	};

	VolumeRaycaster(
		std::shared_ptr<Shader> shaderPosition,
		std::shared_ptr<Shader> shaderRaycast,
		std::shared_ptr<Shader> shaderSlice
	);

	
	bool setVolume(const blib::Volume & volume, int channel);

	void render(
		const Camera & camera,
		ivec4 viewport,
		Shader & shaderPosition,
		Shader & shaderRaycast
	);

	void renderSlice(int axis, ivec2 screenPos, ivec2 screenSize) const;

	vec3 sliceMin;
	vec3 sliceMax;

	float opacityWhite;
	float opacityBlack;

	bool preserveAspectRatio;
	bool showGradient;

	void setTransferJet();
	void setTransferGray();

private:
	EnterExitVolume _enterExit;

	
	Texture _transferTexture;

	GLuint _volTexture;
	ivec3 _volDim;

	

	VertexBuffer<VertexData> _cube;
	VertexBuffer<VertexData> _quad;

	std::shared_ptr<Shader> _shaderPosition;
	std::shared_ptr<Shader> _shaderRaycast;
	std::shared_ptr<Shader> _shaderSlice;

};