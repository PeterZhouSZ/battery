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
		ivec4 viewport
	);

	void renderSlice(int axis, ivec2 screenPos, ivec2 screenSize) const;

	void renderGrid(const Camera & camera, ivec4 viewport, Shader & shader, float opacity = 0.1f);

	vec3 sliceMin;
	vec3 sliceMax;

	float opacityWhite;
	float opacityBlack;

	bool preserveAspectRatio;
	bool showGradient;

	

	void setTransferJet();
	void setTransferGray();

	void enableFiltering(bool val);

private:
	EnterExitVolume _enterExit;

	
	Texture _transferTexture;

	GLuint _volTexture;
	ivec3 _volDim;
	PrimitiveType _volType;
	bool _enableFiltering;
	

	VertexBuffer<VertexData> _cube;
	VertexBuffer<VertexData> _quad;

	std::shared_ptr<Shader> _shaderPosition;
	std::shared_ptr<Shader> _shaderRaycast;
	std::shared_ptr<Shader> _shaderSlice;

};