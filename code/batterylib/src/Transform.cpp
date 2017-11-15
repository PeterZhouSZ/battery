#include "Transform.h"

using namespace Eigen;

Affine3f blib::Transform::getAffine() const
{
	return Translation3f(translation) * rotation * Scaling(scale[0], scale[1], scale[2]);
}

Affine3f blib::Transform::getInverseAffine() const
{
	//todo: use conjugate quat?
	return 
		Scaling(1.0f / scale[0],  1.0f / scale[1], 1.0f / scale[2]) 
		*  rotation.inverse() 
		* Translation3f(-translation);
}

Eigen::Matrix3f blib::Transform::getRotation() const
{
	return rotation.toRotationMatrix();
}

Eigen::Vector3f blib::Transform::applyToPointInverse(const Eigen::Vector3f & point) const
{
	return getInverseAffine() * point;
}

