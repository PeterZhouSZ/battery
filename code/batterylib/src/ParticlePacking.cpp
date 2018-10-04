#include "ParticlePacking.h"


namespace blib {

	BLIB_EXPORT blib::ConvexPolyhedronParticle::ConvexPolyhedronParticle(const std::shared_ptr<ConvexPolyhedron> & shape, const Transform & transform) :
		_shape(shape)
	{
		setTransform(transform);
	}

	BLIB_EXPORT void ConvexPolyhedronParticle::setTransform(const Transform & t)
	{
		_transform = t;
		_shapeInstance = _shape->transformed(getTransform());
	}

	BLIB_EXPORT AABB ConvexPolyhedronParticle::bounds() const
	{
		return _shapeInstance.bounds();
	}

	BLIB_EXPORT bool ConvexPolyhedronParticle::intersects(const Particle & other) const
	{

		const ConvexPolyhedronParticle* same = dynamic_cast<const ConvexPolyhedronParticle*>(&other);
		if (same != nullptr) {
			return _shapeInstance.intersects(same->_shapeInstance);
		}
		else {
			assert(false);
			return false;
		}


	}

}