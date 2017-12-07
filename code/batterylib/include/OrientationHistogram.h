#include "BatteryLibDef.h"

#include <vector>
#include <Eigen/Eigen>

#include "RandomGenerator.h"

namespace blib {


	struct OrientHistogram {
		BLIB_EXPORT OrientHistogram(
			unsigned int bucketsYaw, 
			unsigned int bucketsPitch,
			unsigned int bucketsRoll
		);

		/*
			orientation in euler angles
		*/
		BLIB_EXPORT void add(const Eigen::Vector3f & orientation);
		BLIB_EXPORT void add(float yaw, float pitch, float roll);

		BLIB_EXPORT Eigen::Vector3i size() const;
		BLIB_EXPORT const std::vector<size_t> & counts() const;
		
	private:
		std::vector<size_t> _counts;		
		const Eigen::Vector3i _size;
		const Eigen::Vector3f _sizeF;
	};

	struct OrientDistribution {
		BLIB_EXPORT OrientDistribution(const OrientHistogram & hist);

		BLIB_EXPORT float getMRD(const Eigen::Vector3f & orientation) const;
		BLIB_EXPORT float getMRD(float yaw, float pitch, float roll) const;
	private:
		std::vector<float> _values;
		const Eigen::Vector3i _size;
		const Eigen::Vector3f _sizeF;	
		const float _totalValues;
	};

	/*
		Returns a vector that is (MRD) times likely to be aligned within deltaRad to axis than uniformly distributed.
	*/
	BLIB_EXPORT Eigen::Vector3f randomOrientation(
		RNGUniformFloat & rnd,
		float MRD,
		const Eigen::Vector3f & axis,
		float deltaRad,
		bool symmetric = true
	);
	

}