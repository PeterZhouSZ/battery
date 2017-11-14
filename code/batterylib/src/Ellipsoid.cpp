#include "Ellipsoid.h"

using namespace blib;

const float pi = static_cast<float>(std::acos(-1.0));

float blib::Ellipsoid::c() const { return param[0]; }
float &blib::Ellipsoid::c() { return param[2]; }

float blib::Ellipsoid::a() const { return param[0]; }
float &blib::Ellipsoid::a() { return param[0]; }

float blib::Ellipsoid::b() const { return param[1]; }
float &blib::Ellipsoid::b() { return param[0]; }

Eigen::Vector3f blib::Ellipsoid::surfacePoint(float theta, float phi) const
{
	return {
		a() * cos(theta) * cos(phi),
		b() * cos(theta) * sin(phi),
		c() * sin(theta)
	};
}

float blib::Ellipsoid::volume() const {
	return (4.0f / 3.0f) * pi * a() * b() * c();
}

float blib::Ellipsoid::surfaceAreaApproximation() const
{
	//https://en.wikipedia.org/wiki/Ellipsoid
	//http://www.numericana.com/answer/ellipsoid.htm#thomsen

	const float p = 1.6075f;

	auto ap = pow(a(), p);
	auto bp = pow(b(), p);
	auto cp = pow(c(), p);

	return 4.0f * pi * pow(
		(ap*bp + ap*cp + bp*cp) / 3.0f,
		1.0f / p
	);

}

bool blib::Ellipsoid::isPointIn(const Eigen::Vector3f &pt) {
  float v = pt[0] * pt[0] / (a() * a()) + pt[1] * pt[1] / (b() * b()) +
            pt[2] * pt[2] / (c() * c());

  return v < 1.0f;
}

bool blib::Ellipsoid::isPointInGlobal(const Eigen::Vector3f &pt) {
	return isPointIn(transform.applyToPointInverse(pt));
}
