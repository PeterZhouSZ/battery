#include "VolumeRaycaster.h"
#include "GLError.h"

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
}
