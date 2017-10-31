#include "Transform.h"

using namespace Eigen;

Affine3f blib::Transform::getAffine() const
{
	return Translation3f(translation) * rotation * Scaling(scale[0], scale[1], scale[2]);
}

