#pragma once

#include "BatteryLibDef.h"

#include <Eigen/Eigen>
#include <array>

namespace blib {
		
	struct Triangle {
		std::array<Eigen::Vector3f, 3> v;		

		/*
			Returns cross product of triangle's edges
		*/
		BLIB_EXPORT Eigen::Vector3f cross() const;

		/*
			Returns unit normal vector of the triangle
		*/
		BLIB_EXPORT Eigen::Vector3f normal() const;

		/*
			Returns area of the triangle
		*/
		
		BLIB_EXPORT Eigen::Vector3f::value_type area() const;

		/*
			Returns triangle with reverse winding order
		*/
		BLIB_EXPORT Triangle flipped() const;

	};

}