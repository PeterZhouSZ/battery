#pragma once


#include "Volume.h"

namespace blib {


	struct GeneratorSphereParams {
		size_t N;
		float rmin;
		float rmax;
		bool overlapping;
		bool withinBounds;
		size_t maxTries;
	};


	BLIB_EXPORT VolumeChannel generateSpheres(ivec3 res, const GeneratorSphereParams & p, bool * sucess);



}