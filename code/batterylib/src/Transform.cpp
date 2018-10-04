#include "Transform.h"

#include <glm/gtc/matrix_transform.hpp>

namespace blib {

	BLIB_EXPORT mat4 blib::Transform::getAffine() const
	{
		return glm::translate(glm::mat4(), translation) * getRotation<mat4>() * glm::scale(glm::mat4(), scale);
	}

	BLIB_EXPORT mat4 blib::Transform::getInverseAffine() const
	{
		return 
			glm::scale(glm::mat4(), vec3(1.0f / scale.x, 1.0f / scale.y, 1.0f / scale.z)) *
			getInverseRotation<mat4>() * 
			glm::translate(glm::mat4(), -translation);
	}

	
}



using namespace Eigen;

Affine3f blib::EigenTransform::getAffine() const
{
	return Translation3f(translation) * rotation * Scaling(scale[0], scale[1], scale[2]);
}

Affine3f blib::EigenTransform::getInverseAffine() const
{
	//todo: use conjugate quat?
	return 
		Scaling(1.0f / scale[0],  1.0f / scale[1], 1.0f / scale[2]) 
		*  rotation.inverse() 
		* Translation3f(-translation);
}

Eigen::Matrix3f blib::EigenTransform::getRotation() const
{
	return rotation.toRotationMatrix();
}

Eigen::Vector3f blib::EigenTransform::applyToPointInverse(const Eigen::Vector3f & point) const
{
	return getInverseAffine() * point;
}

