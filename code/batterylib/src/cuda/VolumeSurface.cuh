#pragma once

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>

#include "Volume.cuh"


/*
	Calculates area for each voxel
*/

//Simple voxel edge count 
void countVolumeInterface(	
	uint3 res,
	PrimitiveType volumeType,
	const cudaSurfaceObject_t volume,
	PrimitiveType countType,
	cudaSurfaceObject_t count
);

//Marching cubes
void countVolumeInterface_MarchingCubes(
	uint3 res,
	PrimitiveType volumeType,
	const cudaSurfaceObject_t volume,
	PrimitiveType countType,
	cudaSurfaceObject_t count
);

void countVolumeInterface_MarchingCubes_Smoothed(
	uint3 res,
	PrimitiveType volumeType,
	const cudaSurfaceObject_t volume,
	PrimitiveType countType,
	cudaSurfaceObject_t count,
	int kernelSize
);


void sumOverVolume(	
	uint3 res,
	PrimitiveType type,
	const cudaSurfaceObject_t volume,
	void * auxBufferGPU,
	void * auxBufferCPU,
	void * result
);