#pragma once

#include "BatteryLibDef.h"
#include "Transform.h"
#include "RandomGenerator.h"
#include "AABB.h"

#include <Eigen/Eigen>

namespace blib {


	/*
		scaling of 3 axes by a,b,c (x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1)
	*/
	using EllipsoidParam = Eigen::Vector3f;


	struct Ellipsoid {

		static const float theta_min;
		static const float theta_max;
		static const float theta_range;
		static const float phi_min;
		static const float phi_max;
		static const float phi_range;
		
		Transform transform;

		BLIB_EXPORT float & a();
		BLIB_EXPORT float & b();
		BLIB_EXPORT float & c();

		BLIB_EXPORT float a() const;
		BLIB_EXPORT float b() const;
		BLIB_EXPORT float c() const;

		BLIB_EXPORT Eigen::Vector3f surfacePoint(float theta, float phi) const;

		BLIB_EXPORT float volume() const;

		BLIB_EXPORT float surfaceAreaApproximation() const;

		bool isOblate(const float eps) const;
		bool isProlate(const float eps) const;

		/*
			Point has to be in ellipsoids space (inverse of transform)
		*/
		BLIB_EXPORT bool isPointIn(const Eigen::Vector3f & pt) const;


		BLIB_EXPORT bool isPointInGlobal(const Eigen::Vector3f & pt) const;

			
		BLIB_EXPORT AABB aabb() const;

	};

	

	BLIB_EXPORT bool ellipsoidEllipsoidMonteCarlo(
		const Ellipsoid & a, 
		const Ellipsoid & b, 
		RNGUniformFloat & randomGenerator,
		int sampleCount = 16
		);



}