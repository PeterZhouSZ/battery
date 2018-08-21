#pragma once

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>

#include "Volume.cuh"


struct MGGPU_Volume{
	uint3 res;	
	PrimitiveType type;
	cudaSurfaceObject_t surf;
};

template <size_t size>
struct MGGPU_Kernel3D {
	double v[size][size][size];
};

using MGGPU_InterpKernel = MGGPU_Kernel3D<2>;
using MGGPU_RestrictKernel = MGGPU_Kernel3D<4>;
using MGGPU_SystemTopKernel = MGGPU_Kernel3D<3>;

using MGGPU_KernelPtr = double *;


inline __device__ __host__ MGGPU_SystemTopKernel MGGPU_GetSystemKernel(
	const MGGPU_Volume & domain,
	const uint3 & vox,
	const float3 & cellDim,
	const Dir & dir
) {
	MGGPU_SystemTopKernel kernel;

	return kernel;
}

inline __device__ __host__ MGGPU_InterpKernel MGGPU_GetInterpolationKernel(
	const MGGPU_Volume & domain,
	const uint3 & vox,
	const Dir & dir
) {
	MGGPU_InterpKernel kernel;

	return kernel;
}

inline __device__ __host__ MGGPU_RestrictKernel MGGPU_GetRestrictionKernel(
	const uint3 & res, //domain size
	const uint3 & vox
){
	MGGPU_RestrictKernel kernel;
	return kernel;
}

inline __device__ __host__ void MGGPU_CombineKernels(
	MGGPU_KernelPtr * a, int an,
	MGGPU_KernelPtr * b, int bn,
	MGGPU_KernelPtr * c
) {
	int cn = (an + bn - 1);		

}


void MGGPU_GenerateDomain(
	const MGGPU_Volume & binaryMask,
	double value_zero,
	double value_one,
	MGGPU_Volume & output
);


void MGGPU_RestrictDomain(
	const MGGPU_Volume & domain,
	MGGPU_Volume & output
);


void MGGPU_Convolve(
	const MGGPU_Volume & in,
	MGGPU_KernelPtr * kernels, int kn,
	const MGGPU_Volume & out
);