#pragma once

#include "BatteryLibDef.h"

namespace blib {


	
	class VolumeChannel;

	BLIB_EXPORT void rasterize(
		float * meshTriangles, size_t triangleN,
		float * transformMatrices4x4, size_t instanceN,
		VolumeChannel & output
	);

}