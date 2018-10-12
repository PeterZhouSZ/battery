#include "GeometryObject.h"
#include "Geometry.h"

namespace blib {

	BLIB_EXPORT blib::GeometryObject::GeometryObject(std::shared_ptr<Geometry> geom) : _templateGeometry(geom)
	{

	}


	void GeometryObject::setTransform(Transform & transform)
	{
		_geomDirty = true;
		_boundsDirty = true;
		_transform = transform;
	}

	BLIB_EXPORT Transform GeometryObject::getTransform() const
	{
		return _transform;
	}

	BLIB_EXPORT AABB GeometryObject::bounds() const
	{
		if (_boundsDirty || _geomDirty) {
			_bounds = getGeometry()->bounds();
			_boundsDirty = false;
		}
		return _bounds;
	}

	BLIB_EXPORT const std::unique_ptr<blib::Geometry> & GeometryObject::getGeometry() const
	{
		if (_geomDirty) {
			_geometryCached = _templateGeometry->transformed(_transform);
			_geomDirty = false;
		}
		return _geometryCached;
	}

	BLIB_EXPORT const std::shared_ptr<blib::Geometry> & GeometryObject::getTemplateGeometry() const
	{
		return _templateGeometry;
	}

}
