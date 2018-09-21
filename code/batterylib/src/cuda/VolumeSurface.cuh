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
	const CUDA_Volume & input,
	CUDA_Volume & countOutput
);

//Marching cubes
void countVolumeInterface_MarchingCubes(
	const CUDA_Volume & input,
	CUDA_Volume & countOutput
);

void countVolumeInterface_MarchingCubes_Smoothed(
	const CUDA_Volume & input,
	CUDA_Volume & countOutput,
	int kernelSize
);

void sumOverVolume(	
	const CUDA_Volume & input,
	void * auxBufferGPU,
	void * auxBufferCPU,
	void * result
);