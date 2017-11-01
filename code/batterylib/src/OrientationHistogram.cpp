#include "OrientationHistogram.h"

#include <unsupported/Eigen/EulerAngles>
#include <numeric>

using namespace Eigen;

const float pi = static_cast<float>(std::acos(-1.0));
const Vector3f MinEuler = -Vector3f(pi,pi,pi);
const Vector3f MaxEuler = -MinEuler;
const Vector3f EulerSpanInv = (MaxEuler - MinEuler).cwiseInverse();


blib::OrientHistogram::OrientHistogram(unsigned int bucketsYaw, unsigned int bucketsPitch, unsigned int bucketsRoll)
	: _counts(bucketsYaw*bucketsPitch*bucketsYaw, 0), 
	_size{bucketsYaw,bucketsPitch,bucketsRoll},
	_sizeF{ static_cast<float>(bucketsYaw), static_cast<float>(bucketsPitch), static_cast<float>(bucketsRoll)}
{
	
}

void blib::OrientHistogram::add(const Eigen::Vector3f & orientation)
{

	const Vector3i indices = (orientation - MinEuler).cwiseProduct(EulerSpanInv).cwiseProduct(_sizeF).cast<int>().cwiseMin(_size - Vector3i{1,1,1});
	const size_t index = indices[0] + indices[1] * _size[0] + indices[2] * _size[0] * _size[1];
	
	++_counts[index];
}

Eigen::Vector3i blib::OrientHistogram::size() const
{
	return _size;
}

const std::vector<size_t> & blib::OrientHistogram::counts() const
{
	return _counts;
}

void blib::OrientHistogram::add(float yaw, float pitch, float roll)
{
	add({ yaw,pitch,roll });
}

/********************************/

blib::OrientDistribution::OrientDistribution(const OrientHistogram & hist) :
	_values(hist.counts().size(), 0.0f),
	_size(hist.size()),
	_sizeF(hist.size().cast<float>()),
	_totalValues(static_cast<float>(hist.counts().size()))
{

	size_t totalCnt = std::accumulate(hist.counts().begin(), hist.counts().end(), size_t(0));
	float totalCntF = static_cast<float>(totalCnt);

	const auto & counts = hist.counts();	
	for (auto i = 0; i < hist.counts().size(); i++) {
		_values[i] = counts[i] / totalCntF;
	}

}

float blib::OrientDistribution::getMRD(const Eigen::Vector3f & orientation) const
{
	const Vector3i indices = (orientation - MinEuler).cwiseProduct(EulerSpanInv).cwiseProduct(_sizeF).cast<int>();
	const size_t index = indices[0] + indices[1] * _size[0] + indices[2] * _size[0] * _size[1];
	return /*1.0f /*/ (_values[index] * _totalValues);
}

float blib::OrientDistribution::getMRD(float yaw, float pitch, float roll) const
{
	return getMRD({ yaw,pitch,roll });
}


