#pragma once

#include "BatteryLibDef.h"
#include <vector>
#include <memory>

namespace blib {


	
	class VolumeChannel;
	struct GeometryObject;


	BLIB_EXPORT void rasterize(
		const std::vector<std::shared_ptr<GeometryObject>> & objects,
		VolumeChannel & output
	);

	BLIB_EXPORT void rasterize(
		const float * meshTriangles, size_t triangleN,
		const float * transformMatrices4x4, size_t instanceN,
		VolumeChannel & output
	);

}