#include "Scene.h"


bool SceneObject::_isValid() const
{
	return _valid;
}

const VertexBuffer<VertexData> & SceneObject::getVBO() const
{
	if (!_isValid()) {
		_valid = _updateBuffer();
		if (!_valid)
			throw "Failed to update gpu buffer";
	}
	return _buffer;
}

void SceneObject::_invalidate()
{
	_valid = false;
}

