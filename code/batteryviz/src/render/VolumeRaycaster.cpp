#include "VolumeRaycaster.h"
#include "GLGlobal.h"

#include "render/PrimitivesVBO.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace blib;

void VolumeRaycaster::EnterExitVolume::resize(GLuint w, GLuint h) {
  if (w != enterTexture.size.x || h != enterTexture.size.y) {
    enterTexture.size = {w, h, 0};
    GL(glBindFramebuffer(GL_FRAMEBUFFER, enterFramebuffer.ID()));
    GL(glBindTexture(GL_TEXTURE_2D, enterTexture.ID()));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, NULL));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, enterTexture.ID(), 0));
    GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  }
  if (w != exitTexture.size.x || h != exitTexture.size.y) {
    exitTexture.size = {w, h, 0};
    GL(glBindFramebuffer(GL_FRAMEBUFFER, exitFramebuffer.ID()));
    GL(glBindTexture(GL_TEXTURE_2D, exitTexture.ID()));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, NULL));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, exitTexture.ID(), 0));
    GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  }
}

VolumeRaycaster::VolumeRaycaster(std::shared_ptr<Shader> shaderPosition,
                                 std::shared_ptr<Shader> shaderRaycast,
                                 std::shared_ptr<Shader> shaderSlice)
    : _shaderPosition(std::move(shaderPosition)),
      _shaderRaycast(std::move(shaderRaycast)),
      _shaderSlice(std::move(shaderSlice)), _cube(getCubeVBO()),
      _quad(getQuadVBO()), sliceMin({-1, -1, -1}), sliceMax({1, 1, 1}),
      _volumeTexture(GL_TEXTURE_3D, 0, 0, 0), showGradient(false) {

  {
    std::vector<color4> transferVal(16);
    for (auto &v : transferVal)
      v = color4(0.5f);

    transferVal[15] = vec4(0.0f);

    _transferTexture = Texture(GL_TEXTURE_1D, 16, 1, 0);
    glBindTexture(GL_TEXTURE_1D, _transferTexture.ID());
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA,
                 static_cast<GLsizei>(transferVal.size()), 0, GL_RGBA, GL_FLOAT,
                 transferVal.data());
    glBindTexture(GL_TEXTURE_1D, 0);
  }
}

bool VolumeRaycaster::updateVolume(const Volume<unsigned char> &volume) {
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  GL(glBindTexture(GL_TEXTURE_3D, _volumeTexture.ID()));
  auto dims = volume.dimensions();

  GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, static_cast<GLsizei>(dims[0]),
                  static_cast<GLsizei>(dims[1]), static_cast<GLsizei>(dims[2]),
                  0, GL_RED, GL_UNSIGNED_BYTE, volume.data()));
  GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER,
                     GL_LINEAR_MIPMAP_LINEAR));

  GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
  GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));

  // GL(glGenerateMipmap(GL_TEXTURE_3D));

  GL(glBindTexture(GL_TEXTURE_3D, 0));

  _volumeTexture.size = {dims[0], dims[1], dims[2]};

  return true;
}

void VolumeRaycaster::render(const Camera &camera, ivec4 viewport,
                             Shader &shaderPosition, Shader &shaderRaycast) {

  _enterExit.resize(viewport[2], viewport[3]);

  // Render enter/exit texture
  {
    auto &shader = shaderPosition;

    mat4 M = mat4(1.0f);
    if (preserveAspectRatio) {

      // include bug for max?
      int maxDim = (_volumeTexture.size.x > _volumeTexture.size.y)
                       ? _volumeTexture.size.x
                       : _volumeTexture.size.y;
      maxDim =
          (maxDim > _volumeTexture.size.z) ? maxDim : _volumeTexture.size.z;

      vec3 scale =
          vec3(_volumeTexture.size) * (1.0f / static_cast<float>(maxDim));

      M = glm::scale(mat4(), scale);
    }

    shader.bind();
    shader["PVM"] = camera.getPV() * M;
    shader["minCrop"] = sliceMin;
    shader["maxCrop"] = sliceMax;

    GL(glBindFramebuffer(GL_FRAMEBUFFER, _enterExit.enterFramebuffer.ID()));
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glViewport(0, 0, viewport[2], viewport[3]);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    _cube.render();

    GL(glBindFramebuffer(GL_FRAMEBUFFER, _enterExit.exitFramebuffer.ID()));
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glViewport(0, 0, viewport[2], viewport[3]);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    _cube.render();

    GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    shader.unbind();
  }

  // Raycast
  {
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glDisable(GL_CULL_FACE);

    auto &shader = shaderRaycast;

    shader.bind();
    shader["transferFunc"] = _transferTexture.bindTo(GL_TEXTURE0);
    shader["volumeTexture"] = _volumeTexture.bindTo(GL_TEXTURE1);
    shader["enterVolumeTex"] = _enterExit.enterTexture.bindTo(GL_TEXTURE2);
    shader["exitVolumeTex"] = _enterExit.exitTexture.bindTo(GL_TEXTURE3);

    shader["steps"] = 128;
    shader["transferOpacity"] = 0.5f;
    shader["blackOpacity"] = opacityBlack;
    shader["whiteOpacity"] = opacityWhite;

    shader["resolution"] =
        vec3(1.0f / _volumeTexture.size.x, 1.0f / _volumeTexture.size.y,
             1.0f / _volumeTexture.size.z);
    shader["showGradient"] = showGradient;

    shader["viewPos"] = camera.getPosition();

    _quad.render();

    shader.unbind();
  }
}

void VolumeRaycaster::renderSlice(int axis, ivec2 screenPos,
                                  ivec2 screenSize) const {
  GL(glDisable(GL_CULL_FACE));
  GL(glViewport(screenPos.x, screenPos.y, screenSize.x, screenSize.y));
  auto &shader = *_shaderSlice;

  float t = sliceMin[axis];
  if (axis == 0) {
    t = sliceMin[2];
  } else if (axis == 1) {
    t = sliceMin[1];
  } else if (axis == 2) {
    t = sliceMin[0];
  }

  shader.bind();

  // Set uniforms
  shader["tex"] = _volumeTexture.bindTo(GL_TEXTURE0);
  shader["slice"] = (t + 1) / 2.0f;
  shader["axis"] = axis;

  // Render fullscreen quad
  _quad.render();

  shader.unbind();
}
