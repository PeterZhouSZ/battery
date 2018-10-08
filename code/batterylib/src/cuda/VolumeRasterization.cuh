#pragma once

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>

#include "Volume.cuh"


void Volume_Rasterize(
	float * meshTriangles, size_t triangleN,
	float * transformMatrices4x4, size_t instanceN,
	CUDA_Volume & output
);