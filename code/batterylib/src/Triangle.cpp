#include "Triangle.h"

using namespace Eigen;

Eigen::Vector3f blib::Triangle::cross() const
{
	const Vector3f e[2] = {
		v[1] - v[0],
		v[2] - v[0]
	};
	return e[0].cross(e[1]);
}

Vector3f blib::Triangle::normal() const
{	
	return cross().normalized();
}

Eigen::Vector3f::value_type blib::Triangle::area() const
{
	return cross().norm() * 0.5f;
}

blib::Triangle blib::Triangle::flipped() const
{
	return { v[0],v[2],v[1] };
}

