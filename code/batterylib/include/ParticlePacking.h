#pragma once

#include <vector>

#include "Transform.h"
#include "ConvexPolyhedron.h"
#include "AABB.h"

namespace blib {


	
	using Tree = int;
	

	class Particle {
		
	public:
		virtual AABB bounds() const = 0;
		virtual bool intersects(const Particle & other) const = 0;

		virtual void setTransform(const Transform & t) = 0;

		Transform getTransform() const { return _transform;	}

		virtual void * getTemplateShapeAddress() { return nullptr; }

	protected:
		Transform _transform;
		
	};

	class ConvexPolyhedronParticle : public Particle {

	public:
		BLIB_EXPORT ConvexPolyhedronParticle(const std::shared_ptr<ConvexPolyhedron> & shape, const Transform & transform);
		
		BLIB_EXPORT void setTransform(const Transform & t);

		BLIB_EXPORT AABB bounds() const;

		BLIB_EXPORT bool intersects(const Particle & other) const;

		BLIB_EXPORT virtual void * getTemplateShapeAddress() { return _shape.get(); }

	private:
		std::shared_ptr<ConvexPolyhedron> _shape;
		ConvexPolyhedron _shapeInstance; // temp for collison detection
	};

	

	class ParticlePacking {

	public:
		std::vector<
			std::shared_ptr<Particle>
		> particles;

		//Tree tree;

	};


}