#pragma once

#include <cuda_runtime.h>

//https://stackoverflow.com/questions/12778949/cuda-memory-alignment
#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

struct MGGPU_Volume{
	uint3 res;	
	PrimitiveType type;
	cudaSurfaceObject_t surf;	
	int volID;

	void * cpu; //for debug
};


template <size_t size>
struct MGGPU_Kernel3D {
	double v[size][size][size];
};

//7-point stencil
struct MY_ALIGN(16) MGGPU_SystemTopKernel {
	double v[7];
};

#define INTERP_SIZE 3
#define RESTR_SIZE 4
#define DOMAIN_RESTR_SIZE 2

using MGGPU_InterpKernel = MGGPU_Kernel3D<INTERP_SIZE>;
using MGGPU_RestrictKernel = MGGPU_Kernel3D<RESTR_SIZE>;
using MGGPU_DomainRestrictKernel = MGGPU_Kernel3D<DOMAIN_RESTR_SIZE>;
using MGGPU_KernelPtr = double *;

//Construction params
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

struct MGGPU_SmootherParams {
	MGGPU_KernelPtr A;
	bool isTopLevel;
	Dir dir;
	
	MGGPU_Volume f;
	MGGPU_Volume x;
	MGGPU_Volume tmpx;
	MGGPU_Volume r;
	uint3 res;
	double tolerance;
	void * auxBufferGPU;
	void * auxBufferCPU;
	int iter;
	double alpha; //under/overrelaxation
};