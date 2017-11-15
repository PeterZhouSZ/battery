#pragma once

#include <Eigen/Eigen>

namespace blib {
	struct AABB {
		Eigen::Vector3f min;
		Eigen::Vector3f max;
	};
}