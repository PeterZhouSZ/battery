#include "VolumeRaycaster.h"
#include "GLGlobal.h"

#include "render/PrimitivesVBO.h"

void VolumeRaycaster::EnterExitVolume::resize(GLuint w, GLuint h)
{
	if (w != enterTexture.size.x || h != enterTexture.size.y) {
		enterTexture.size = { w, h, 0};
		GL(glBindFramebuffer(GL_FRAMEBUFFER, enterFramebuffer.ID()));
		GL(glBindTexture(GL_TEXTURE_2D, enterTexture.ID()));
		GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, NULL));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, enterTexture.ID(), 0));
		GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	}
	if (w != exitTexture.size.x || h != exitTexture.size.y) {
		exitTexture.size = { w, h, 0 };
		GL(glBindFramebuffer(GL_FRAMEBUFFER, exitFramebuffer.ID()));
		GL(glBindTexture(GL_TEXTURE_2D, exitTexture.ID()));
		GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, NULL));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, exitTexture.ID(), 0));
		GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	}
}

VolumeRaycaster::VolumeRaycaster(
	std::shared_ptr<Shader> shaderPosition,
	std::shared_ptr<Shader> shaderRaycast,
	std::shared_ptr<Shader> shaderSlice
): 
	_shaderPosition(std::move(shaderPosition)),
	_shaderRaycast(std::move(shaderRaycast)),
	_shaderSlice(std::move(shaderSlice)),
	_cube(getCubeVBO()),
	_quad(getQuadVBO()),
	_sliceMin({ -1,-1,-1 }),
	_sliceMax({ 1,1,1 }),
	_volumeTexture(GL_TEXTURE_3D, 0,0,0)
{
	
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

}

bool VolumeRaycaster::updateVolume(const Volume<unsigned char> & volume)
{
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	GL(glBindTexture(GL_TEXTURE_3D, _volumeTexture.ID()));
	GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, volume.size.x, volume.size.y, volume.size.z, 0, GL_RED, GL_UNSIGNED_BYTE, volume.data.data()));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
	GL(glBindTexture(GL_TEXTURE_3D, 0));

	_volumeTexture.size = volume.size;

	return true;
}


void VolumeRaycaster::render(
	const Camera & camera, 
	ivec4 viewport, 
	Shader & shaderPosition,
	Shader & shaderRaycast
)
{
	
	
	_enterExit.resize(viewport[2], viewport[3]);

	//Render enter/exit texture
	{	
		auto & shader = shaderPosition;

		shader.bind();
		shader["PVM"] = camera.getPV();
		shader["minCrop"] = _sliceMin;
		shader["maxCrop"] = _sliceMax;

		GL(glBindFramebuffer(GL_FRAMEBUFFER, _enterExit.enterFramebuffer.ID()));
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		glViewport(0, 0, viewport[2], viewport[3]);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_FRONT);
		_cube.render();

		GL(glBindFramebuffer(GL_FRAMEBUFFER, _enterExit.exitFramebuffer.ID()));
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		glViewport(0, 0, viewport[2], viewport[3]);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		_cube.render();

		GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

		shader.unbind();

	}

	//Raycast
	{
		glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
		glDisable(GL_CULL_FACE);

		auto & shader = shaderRaycast;

		shader.bind();
		shader["transferFunc"] = _transferTexture.bindTo(GL_TEXTURE0);
		shader["volumeTexture"] = _volumeTexture.bindTo(GL_TEXTURE1);
		shader["enterVolumeTex"] = _enterExit.enterTexture.bindTo(GL_TEXTURE2);
		shader["exitVolumeTex"] = _enterExit.exitTexture.bindTo(GL_TEXTURE3);

		shader["steps"] = 128;
		shader["transferOpacity"] = 0.5f;
		shader["blackOpacity"] = 0.004f; //todo
		shader["whiteOpacity"] = 0.05f;


		_quad.render();

		shader.unbind();

	}

}

void VolumeRaycaster::renderSlice(int axis, ivec2 screenPos, ivec2 screenSize) const
{
	GL(glDisable(GL_CULL_FACE));
	GL(glViewport(screenPos.x, screenPos.y, screenSize.x, screenSize.y));
	auto &shader = *_shaderSlice;

	const float t = _sliceMin[axis];

	shader.bind();

	//Set uniforms
	shader["tex"] = _volumeTexture.bindTo(GL_TEXTURE0);
	shader["slice"] = (t + 1) / 2.0f;
	shader["axis"] = axis;

	//Render fullscreen quad
	_quad.render();

	shader.unbind();
}

