#pragma once

#include <cuda_runtime.h>
#include "VolumeTypes.cuh"


#define INTERP_SIZE 3
#define RESTR_SIZE 4
#define DOMAIN_RESTR_SIZE 2

using MGGPU_InterpKernel = CUDA_Kernel3DD<INTERP_SIZE>;
using MGGPU_RestrictKernel = CUDA_Kernel3DD<RESTR_SIZE>;
using MGGPU_DomainRestrictKernel = CUDA_Kernel3DD<DOMAIN_RESTR_SIZE>;




struct MGGPU_SmootherParams {
	CUDA_KernelPtrD A;
	bool isTopLevel;
	Dir dir;
	
	CUDA_Volume f;
	CUDA_Volume x;
	CUDA_Volume tmpx;
	CUDA_Volume r;
	uint3 res;
	double tolerance;
	void * auxBufferGPU;
	void * auxBufferCPU;
	int iter;
	double alpha; //under/overrelaxation
};