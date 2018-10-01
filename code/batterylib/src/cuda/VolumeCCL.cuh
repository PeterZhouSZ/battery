#pragma once

#include "Volume.cuh"





/*
	Outputs labeled volume 
	Returns N - number of labels
	(0 is background, 1...N are labels)
*/
uint VolumeCCL(
	const CUDA_Volume & input,
	CUDA_Volume & output,
	uchar background
);

void VolumeCCL_Colorize(
	const CUDA_Volume & input, //TYPE_UINT	
	CUDA_Volume & output //TYPE_FLOAT3
);