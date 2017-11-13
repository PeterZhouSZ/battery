#pragma once

#include "render/Scene.h"
#include "render/Shaders.h"

#include <batterylib/include/TriangleMesh.h>

class MeshObject : SceneObject {

public:
	const blib::TriangleMesh & getMesh() const;

	blib::TriangleMesh getMesh();

	

	virtual ShaderOptions getShaderOptions(
		ShaderType shaderType,
		const Camera & cam, 
		mat4 parentTransform
	) const override;

protected:
	blib::TriangleMesh _mesh;

	virtual bool _updateBuffer() const override;


};