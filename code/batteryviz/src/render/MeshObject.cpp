#include "MeshObject.h"

#include "Camera.h"

bool MeshObject::_updateBuffer() const {
  std::vector<VertexData> data;
  data.reserve(_mesh.size() * 3);

  VertexData vd;
  vd.color[0] = 0.5f;
  vd.color[1] = 0.5f;
  vd.color[2] = 0.5f;
  vd.color[3] = 1.0f;
  vd.uv[0] = 0.0f;
  vd.uv[1] = 0.0f;

  for (auto t : _mesh) {
    const auto N = t.normal();
    memcpy(&vd.normal, N.data(), N.SizeAtCompileTime * sizeof(float));

    for (auto v : t.v) {
      memcpy(&vd.pos, v.data(), v.SizeAtCompileTime * sizeof(float));
      data.push_back(vd);
    }
  }

  _buffer.setPrimitiveType(GL_TRIANGLES);

  return _buffer.setData(data.begin(), data.end());
}

const blib::TriangleMesh &MeshObject::getMesh() const { return _mesh; }

blib::TriangleMesh MeshObject::getMesh() {
  _invalidate();
  return _mesh;
}

ShaderOptions MeshObject::getShaderOptions(ShaderType shaderType,
                                           const Camera &cam,
                                           mat4 parentTransform) const {

  // Any shaderype, same

  auto M = parentTransform * getTransform();
  auto NM = mat4(glm::inverse(glm::transpose(glm::mat3(M))));

  return { {"M", M}, {"NM", NM}, {"PV", cam.getPV()}, {"viewPos", cam.getPosition() } };
}