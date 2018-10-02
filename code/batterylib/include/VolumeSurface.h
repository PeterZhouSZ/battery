#pragma once

#include "BatteryLibDef.h"
#include "Types.h"


namespace blib {
		
	class VolumeChannel;
	

	BLIB_EXPORT VolumeChannel getVolumeArea(
		const VolumeChannel & mask, 
		ivec3 res = ivec3(0,0,0),
		float isovalue = 0.5f
	);

	BLIB_EXPORT bool getVolumeAreaMesh(
		const VolumeChannel & mask,		
		uint * vboOut, 
		size_t * NvertsOut,
		ivec3 res = ivec3(0, 0, 0),
		float isovalue = 0.5f
	);



}