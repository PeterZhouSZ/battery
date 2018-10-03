#pragma once


#include "Volume.h"
#include <vector>

namespace blib {


	struct GeneratorSphereParams {
		size_t N;
		float rmin;
		float rmax;
		bool overlapping;
		bool withinBounds;
		size_t maxTries;
	};


	struct Sphere {
		vec3 pos;
		float r;
		float r2;
	};


	BLIB_EXPORT std::vector<Sphere> generateSpheres(const GeneratorSphereParams & p);

	BLIB_EXPORT double spheresAnalyticTortuosity(const GeneratorSphereParams & p, const std::vector<Sphere> & spheres);

	BLIB_EXPORT VolumeChannel rasterizeSpheres(ivec3 res, const std::vector<Sphere> & spheres);

	BLIB_EXPORT VolumeChannel generateFilledVolume(ivec3 res, uchar value);



}