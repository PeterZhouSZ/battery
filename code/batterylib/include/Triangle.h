#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#include <array>

namespace blib {
		
	struct Triangle {
		std::array<vec3, 3> v;		

		/*
			Returns cross product of triangle's edges
		*/
		BLIB_EXPORT vec3 cross() const;

		/*
			Returns unit normal vector of the triangle
		*/
		BLIB_EXPORT vec3 normal() const;

		/*
			Returns area of the triangle
		*/
		
		BLIB_EXPORT float area() const;

		/*
			Returns triangle with reverse winding order
		*/
		BLIB_EXPORT Triangle flipped() const;

	};

}