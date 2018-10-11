#pragma once

#include "BatteryLibDef.h"
#include "Transform.h"
#include "AABB.h"

#include <memory>
#include "Geometry.h"

namespace blib {

	//struct Geometry;

	/*
		Geometric object with transform
		Keeps cached transformed geometry & bounds
	*/
	struct GeometryObject {

		BLIB_EXPORT GeometryObject(std::shared_ptr<Geometry> templateGeometry);

		BLIB_EXPORT void setTransform(Transform & transform);

		BLIB_EXPORT Transform getTransform() const;

		BLIB_EXPORT AABB bounds() const;

		BLIB_EXPORT const std::unique_ptr<Geometry> & getGeometry() const;

		BLIB_EXPORT const std::shared_ptr<Geometry> & getTemplateGeometry() const;


	private:
		std::shared_ptr<Geometry> _templateGeometry;
		Transform _transform;

		mutable bool _boundsDirty = true;
		mutable AABB _bounds;

		mutable bool _geomDirty = true;
		mutable std::unique_ptr<Geometry> _geometryCached;
	};

}