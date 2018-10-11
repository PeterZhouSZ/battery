#include "Triangle.h"

namespace blib {

	vec3 blib::Triangle::cross() const
	{
		const vec3 e[2] = {
			v[1] - v[0],
			v[2] - v[0]
		};
		return glm::cross(e[0], e[1]);
	}

	vec3 blib::Triangle::normal() const
	{
		return glm::normalize(cross());
	}

	float blib::Triangle::area() const
	{
		return glm::length(cross()) * 0.5f;
	}

	blib::Triangle blib::Triangle::flipped() const
	{
		return { v[0],v[2],v[1] };
	}

}
