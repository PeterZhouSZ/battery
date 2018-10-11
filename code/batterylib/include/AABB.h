#pragma once

#include "Types.h"
#include <Eigen/Eigen>

namespace blib {
	struct EigenAABB {
		Eigen::Vector3f min;
		Eigen::Vector3f max;
	};

	struct AABB {
		vec3 min = vec3(FLT_MAX);
		vec3 max = vec3(-FLT_MAX);

		AABB getUnion(const AABB & b) {
			AABB c;
			c.min = glm::min(min, b.min);
			c.max = glm::max(max, b.max);
			return c;
		}

		vec3 range() const {
			return max - min;
		}

		vec3 centroid() const {
			return min + range() * 0.5f;
		}

		AABB getIntersection(const AABB & b) const {
			return { glm::max(min, b.min), glm::min(max, b.max) };
		}

		bool isValid() const
		{
			return min.x < max.x && min.y < max.y && min.z < max.z;
		}

		bool testIntersection(const AABB & b) const {
			return getIntersection(b).isValid();
		}

		int largestAxis() const {
			vec3 r = range();
			if (r.x > r.y) {
				if (r.x > r.z) return 0;
				return 2;
			}			
			if (r.y > r.z) return 1;
			return 2;						
		}

	};
}