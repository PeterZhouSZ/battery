#pragma once

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>

#include "Volume.cuh"


struct MGGPU_Volume{
	uint3 res;	
	PrimitiveType type;
	cudaSurfaceObject_t surf;
	int volID;
};



struct MGGPU_SysParams {
	double highConc;
	double lowConc;
	double concetrationBegin;
	double concetrationEnd;
	double cellDim[3];
	double faceArea[3];
	int dirPrimary;
	uint2 dirSecondary;
	Dir dir;
};

bool commitSysParams(const MGGPU_SysParams & sysparams);

template <size_t size>
struct MGGPU_Kernel3D {
	double v[size][size][size];
};

//7-point stencil
struct MY_ALIGN(16) MGGPU_SystemTopKernel {
	double v[7];
};

using MGGPU_InterpKernel = MGGPU_Kernel3D<2>;
using MGGPU_RestrictKernel = MGGPU_Kernel3D<4>;
using MGGPU_DomainRestrictKernel = MGGPU_Kernel3D<2>;
using MGGPU_KernelPtr = double *;


inline __device__ __host__ MGGPU_InterpKernel MGGPU_GetInterpolationKernel(
	const MGGPU_Volume & domain,
	const uint3 & vox,
	const Dir & dir
) {
	MGGPU_InterpKernel kernel;

	return kernel;
}

inline __device__ __host__ MGGPU_DomainRestrictKernel MGGPU_GetDomainRestrictionKernel(){
	MGGPU_DomainRestrictKernel kernel;
	double v = 1.0 / 8.0;
	kernel.v[0][0][0] = v;
	kernel.v[0][0][1] = v;
	kernel.v[0][1][0] = v;
	kernel.v[0][1][1] = v;
	kernel.v[1][0][0] = v;
	kernel.v[1][0][1] = v;
	kernel.v[1][1][0] = v;
	kernel.v[1][1][1] = v;
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

/*
	Convolves using a single kernel
*/
void MGGPU_Convolve(
	const MGGPU_Volume & in,
	MGGPU_KernelPtr kernel, int kn,
	const MGGPU_Volume & out
);


void MGGPU_GenerateSystemTopKernel(
	const MGGPU_Volume & domain,
	MGGPU_SystemTopKernel * A0,
	MGGPU_Volume & f
);