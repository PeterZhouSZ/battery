#pragma once

#include "render/RenderList.h"
#include "render/Shaders.h"

#include <map>


class Camera;


template<typename THost, typename TDevice>
struct HostDeviceResource{
		

private:

	virtual bool _synchronize() = 0;

	bool _validHost;
	bool _validDevice;
	THost _host;
	TDevice _device;
};

struct MeshVBO : public HostDeviceResource<blib::TriangleMesh, VertexData<VertexData>> {

	//implement syncrhonize
};
using VolumeTexture = HostDeviceResource<Volume, Texture>;


class SpatialObject {
	mat4 transform;
};

template <typename T>
class ResourceObject : public SpatialObject {	
	T resource;
};

class MeshObject : public ResourceObject<MeshVBO> {
};

using _Scene = std::map<std::string, SpatialObject>;

/*
	Base class for scene object
	Owns cached gpu resource (vertex buffer)
	Non-const access to data invalides the gpu resource
*/

template <typename T>
class BaseObject {

	mat4 transform;

	const T & getResource() const;

protected:
	virtual bool _updateResource() const = 0;

	bool _isValid() const;
	void _invalidate();

	mutable bool _valid;
	mutable T _resource;
};

class VBOObject : public BaseObject<VertexBuffer<VertexData>> {
		
};

struct Texture;
class VolumeObject : public BaseObject<Texture> {

};


class SceneObject {
	
public:

	const VertexBuffer<VertexData> & getVBO() const;
	
	virtual ShaderOptions getShaderOpt0ions(
		ShaderType shaderType,
		const Camera & cam,
		mat4 parentTransform
	) const = 0;

	mat4 transform;	

protected:
	
	virtual bool _updateBuffer() const = 0;
	
	bool _isValid() const;
	void _invalidate();	

	mutable bool _valid;
	mutable VertexBuffer<VertexData> _buffer;

};

using Scene = std::map<
	std::string,
	std::shared_ptr<SceneObject>
>;

