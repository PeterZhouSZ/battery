#pragma once

#include "BatteryLibDef.h"

#include "Types.h"
#include <glm/gtc/quaternion.hpp>

#include <Eigen/Eigen>

namespace blib {

	struct Transform {
		vec3 translation = {0,0,0};
		glm::quat rotation;
		vec3 scale = {1,1,1};

		BLIB_EXPORT mat4 getAffine() const;
		BLIB_EXPORT mat4 getInverseAffine() const;
		
		template <typename T>
		T getRotation() const {
			if (std::is_same<T, mat4>::value)
				return glm::mat4_cast(rotation);
			if (std::is_same<T, mat3>::value)
				return glm::mat3_cast(rotation);
		}

		template <typename T>
		T getInverseRotation() const {
			if (std::is_same<T, mat4>::value)
				return glm::mat4_cast(glm::inverse(rotation));
			if (std::is_same<T, mat3>::value)
				return glm::mat3_cast(glm::inverse(rotation));
		}
		



	};

	struct EigenTransform {
		Eigen::Vector3f translation = {0,0,0};
		Eigen::Quaternionf rotation = { 0,0,0,1.0f };
		Eigen::Vector3f scale = { 1.0f, 1.0f, 1.0f };

		BLIB_EXPORT Eigen::Affine3f getAffine() const;		
		BLIB_EXPORT Eigen::Affine3f getInverseAffine() const;

		BLIB_EXPORT Eigen::Matrix3f getRotation() const;

		Eigen::Vector3f applyToPointInverse(const Eigen::Vector3f & point) const;

	};

}