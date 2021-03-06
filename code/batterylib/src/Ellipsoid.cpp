#include "Ellipsoid.h"


using namespace blib;

const float pi = static_cast<float>(std::acos(-1.0));



const float blib::Ellipsoid::theta_min = -pi / 2.0f;
const float blib::Ellipsoid::theta_max = pi / 2.0f;
const float blib::Ellipsoid::theta_range = blib::Ellipsoid::theta_max - blib::Ellipsoid::theta_min;
const float blib::Ellipsoid::phi_min = -pi;
const float blib::Ellipsoid::phi_max = pi;
const float blib::Ellipsoid::phi_range = blib::Ellipsoid::phi_max - blib::Ellipsoid::phi_min;

float blib::Ellipsoid::a() const { return transform.scale[0]; }
float &blib::Ellipsoid::a() { return transform.scale[0]; }

float blib::Ellipsoid::b() const { return transform.scale[1]; }
float &blib::Ellipsoid::b() { return transform.scale[1]; }

float blib::Ellipsoid::c() const { return transform.scale[2]; }



float &blib::Ellipsoid::c() { return transform.scale[2]; }

Eigen::Vector3f blib::Ellipsoid::longestAxis() const
{
	if (a() > b()) {
		if (a() > b())
			return a() * Eigen::Vector3f::UnitX();
		else
			return b() * Eigen::Vector3f::UnitY();
	}
	
	if (b() > c())
		return b() * Eigen::Vector3f::UnitY();
	else
		return c() * Eigen::Vector3f::UnitZ();	
	
}

Eigen::Vector3f blib::Ellipsoid::surfacePoint(float theta, float phi) const
{
	return transform.getAffine() * Eigen::Vector3f{
		a() * cos(theta) * cos(phi),
		b() * cos(theta) * sin(phi),
		c() * sin(theta)
	};
}

float blib::Ellipsoid::volume() const {
	return (4.0f / 3.0f) * pi * 
		a() *
		b() *
		c();
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

bool blib::Ellipsoid::isPointIn(const Eigen::Vector3f &pt) const {
	float v = pt[0] * pt[0] / (a() * a()) + pt[1] * pt[1] / (b() * b()) +
            pt[2] * pt[2] / (c() * c());

	return v < 1.0f;
}

bool blib::Ellipsoid::isPointInGlobal(const Eigen::Vector3f &pt) const {
	return isPointIn(transform.applyToPointInverse(pt));
}




blib::EigenAABB blib::Ellipsoid::aabb() const
{
	//https://tavianator.com/exact-bounding-boxes-for-spheres-ellipsoids/

	const auto & t = transform;

	

	const Eigen::Vector3f rsq = t.scale.cwiseProduct(t.scale);
	const auto R = t.getRotation();

	/*const Eigen::Map<const Eigen::RowVector3f> r0(R.data(), 3);
	const Eigen::Map<const Eigen::RowVector3f> r1(R.data() + 3, 3);
	const Eigen::Map<const Eigen::RowVector3f> r2(R.data() + 6, 3);*/	


	
	const Eigen::Vector3f det = {
		sqrt(rsq.dot(R.row(0).cwiseProduct(R.row(0)))),
		sqrt(rsq.dot(R.row(1).cwiseProduct(R.row(1)))),
		sqrt(rsq.dot(R.row(2).cwiseProduct(R.row(2)))),
	};

	return {
		t.translation - det,
		t.translation + det
	};

}

//////////////


bool blib::ellipsoidEllipsoidMonteCarlo(
	const Ellipsoid & a, 
	const Ellipsoid & b, 
	RNGUniformFloat & randomGenerator, 
	int sampleCount /*= 16 */)
{
	const auto phiFunc = [&](){
		return randomGenerator.next() * Ellipsoid::phi_range - Ellipsoid::phi_min;
	};

	const auto thetaFunc = [&]() {
		return randomGenerator.next() * Ellipsoid::theta_range - Ellipsoid::theta_min;
	};

	for (auto i = 0; i < sampleCount; i++) {		
		if (b.isPointInGlobal(a.surfacePoint(thetaFunc(), phiFunc())))
			return true;
	}

	return false;
}
