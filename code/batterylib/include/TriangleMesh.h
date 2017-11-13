#pragma once
#include "Triangle.h"

#include <Eigen/Eigen>
#include <vector>

namespace blib {

	using TriangleMesh = std::vector<Triangle>;

	BLIB_EXPORT TriangleMesh generateSphere(float radius = 1.0f, size_t polarSegments = 16, size_t azimuthSegments = 16);

}