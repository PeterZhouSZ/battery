#pragma once

#include "render/Texture.h"
#include "render/Camera.h"
#include "render/Shader.h"
#include "render/VertexBuffer.h"
#include "render/Framebuffer.h"


#include "batterylib/include/TriangleMesh.h"

#include <unordered_map>
#include <any>
#include <string>
#include <vector>
#include <stack>

enum class ShaderType {
	PHONG
};

using ShaderOptions = std::vector<
	std::pair<std::string, ShaderResourceValue>
>;

struct RenderList {

	

	struct RenderItem {
		const VertexBuffer<VertexData> & vbo;
		const ShaderOptions shaderOptions;
	};

	void clear();

	void add(
		const std::shared_ptr<Shader> & shaderPtr,
		RenderItem item
	);

	void render();



private:
	std::unordered_map<
		std::shared_ptr<Shader>,
		std::vector<RenderItem>
	> _shaderToQueue;
};

/*
	Base class for scene object
	Owns cached gpu resource (vertex buffer)	
	Subclass should define:
		data (mesh, curve, volume)
		shader type
		shader options (as a function of parents transform)
*/
class SceneObject {

	//shader name/ptr?
public:	

	bool isValid() const {
		return _valid;
	}
	
	const VertexBuffer<VertexData> & getVBO() const {
		if (!isValid()) {
			_valid = _updateBuffer();			 
			if (!_valid)
				throw "Failed to update gpu buffer";
		}
		return _buffer;
	}

	virtual ShaderType getShaderType() const = 0;
	virtual ShaderOptions getShaderOptions(
		const Camera & cam,
		mat4 parentTransform
	) const = 0;

	mat4 transform;
	std::vector<std::shared_ptr<SceneObject>> children;

protected:
	virtual bool _updateBuffer() const = 0;
	
	void _invalidate() {
		_valid = false;
	}

	mutable bool _valid;
	mutable VertexBuffer<VertexData> _buffer;
};

//todo map material/type of object to shader enum and shaderoptions



class MeshObject : SceneObject {

public:	
	const blib::TriangleMesh & getMesh() const {
		return _mesh;
	}

	blib::TriangleMesh getMesh() {
		_invalidate();
		return _mesh;
	}

	virtual ShaderType getShaderType() const override;

	virtual ShaderOptions getShaderOptions(const Camera & cam, mat4 parentTransform) const override;

protected:
	blib::TriangleMesh _mesh;
	
	virtual bool _updateBuffer() const override;


};

struct Scene {	
	std::shared_ptr<SceneObject> _root;
};

//stores vbos of scenes objects, if invalidated, updates them
struct SceneRenderer {	
	//volume raycaster? nope, not generic
	//renderlist for meshes
};

void renderSceneObject(
	const std::unordered_map<ShaderType, std::shared_ptr<Shader>> & shaders, 
	const SceneObject & root,
	const Camera & camera
);


