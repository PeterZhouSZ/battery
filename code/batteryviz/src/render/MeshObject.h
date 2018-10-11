#pragma once

#include "render/Scene.h"
#include "render/Shaders.h"

#include <batterylib/include/TriangleMesh.h>

class MeshObject : public SceneObject {

public:

	MeshObject() = default;
	~MeshObject() = default;

	MeshObject(blib::TriangleArray && mesh) : _mesh(mesh), SceneObject() {		
	}

	const blib::TriangleArray & getMesh() const;

	blib::TriangleArray getMesh();

	virtual ShaderOptions getShaderOptions(
		ShaderType shaderType,
		const Camera & cam, 
		mat4 parentTransform
	) const override;

protected:
	blib::TriangleArray _mesh;

	virtual bool _updateBuffer() const override;


};