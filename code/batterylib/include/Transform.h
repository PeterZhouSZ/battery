#pragma once

#include "BatteryLibDef.h"

#include <Eigen/Eigen>

namespace blib {

	struct Transform {
		Eigen::Vector3f translation;
		Eigen::Quaternionf rotation = { 0,0,0,1.0f };
		Eigen::Vector3f scale = { 1.0f, 1.0f, 1.0f };

		BLIB_EXPORT Eigen::Affine3f getAffine() const;		
		BLIB_EXPORT Eigen::Affine3f getInverseAffine() const;

		Eigen::Vector3f applyToPointInverse(const Eigen::Vector3f & point) const;

	};

}