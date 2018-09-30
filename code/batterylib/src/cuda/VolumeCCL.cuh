#pragma once

#include "Volume.cuh"



//Calculates needed dimensions for temporary storage


bool VolumeCCL_Compute(
	const CUDA_Volume & input,
	CUDA_Volume & output,
	uint label = 0
);

