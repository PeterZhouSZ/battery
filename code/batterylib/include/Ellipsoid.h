#pragma once

#include "BatteryLibDef.h"

#include "Transform.h"

#include <Eigen/Eigen>

namespace blib {


	/*
		scaling of 3 axes by a,b,c (x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1)
	*/
	using EllipsoidParam = Eigen::Vector3f;


	struct Ellipsoid {
		EllipsoidParam param;
		Transform transform;

		float & a();
		float & b();
		float & c();

		float a() const;
		float b() const;
		float c() const;

		Eigen::Vector3f surfacePoint(float theta, float phi) const;

		float volume() const;

		float surfaceAreaApproximation() const;

		bool isOblate(const float eps) const;
		bool isProlate(const float eps) const;

		/*
			Point has to be in ellipsoids space (inverse of transform)
		*/
		bool isPointIn(const Eigen::Vector3f & pt);


		bool isPointInGlobal(const Eigen::Vector3f & pt);

		
		

	};

	

}