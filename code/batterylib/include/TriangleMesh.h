#pragma once

#include "Geometry.h"
#include "Triangle.h"
#include <vector>

namespace blib {

	using TriangleArray = std::vector<Triangle>;



	BLIB_EXPORT TriangleArray generateSphere(float radius = 1.0f, size_t polarSegments = 16, size_t azimuthSegments = 16);


	struct TriangleMesh : public Geometry {
		using Edge = ivec2;
		struct Face {
			std::vector<int> vertices;
			vec3 normal;
		};

		std::vector<vec3> vertices;
		std::vector<Face> faces;
		std::vector<Edge> edges;

		BLIB_EXPORT virtual AABB bounds() const override;

		BLIB_EXPORT virtual std::unique_ptr<Geometry> normalized(bool keepAspectRatio = true) const override;

		BLIB_EXPORT virtual std::unique_ptr<Geometry> transformed(const Transform & t) const override;
		
		BLIB_EXPORT void recomputeNormals();		

		BLIB_EXPORT TriangleArray getTriangleArray() const;

	};

}