#pragma once

#include "Types.h"
#include <Eigen/Eigen>

namespace blib {
	struct EigenAABB {
		Eigen::Vector3f min;
		Eigen::Vector3f max;
	};

	struct AABB {
		vec3 min;
		vec3 max;
	};
}