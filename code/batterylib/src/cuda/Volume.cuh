#include <cuda_runtime.h>

#include "CudaMath.h"
#include "PrimitiveTypes.h"



void launchErodeKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut);

void launchHeatKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut);

void launchBinarizeKernel(uint3 res, 
	cudaSurfaceObject_t surfInOut, 
	PrimitiveType type,
	float threshold //between 0 to 1
);



enum Dir {
	X_POS = 0,
	X_NEG = 1,
	Y_POS = 2,
	Y_NEG = 3,
	Z_POS = 4,
	Z_NEG = 5,
	DIR_NONE = 6
};

__host__ __device__ inline int3 dirVec(Dir d) {
	switch (d) {
		case X_POS: return make_int3(1, 0, 0);
		case X_NEG: return make_int3(-1, 0, 0);
		case Y_POS: return make_int3(0, 1, 0);
		case Y_NEG: return make_int3(0, -1, 0);
		case Z_POS: return make_int3(0, 0, 1);
		case Z_NEG: return make_int3(0, 0, -1);
	};
	return make_int3(0, 0, 0);
}

#define BOUNDARY_ZERO_GRADIENT 1e37f

struct DiffuseParams {
	uint3 res;
	cudaSurfaceObject_t mask;
	cudaSurfaceObject_t concetrationIn;
	cudaSurfaceObject_t concetrationOut;
	float zeroDiff;
	float oneDiff;
		
	float boundaryValues[6];
};

void launchDiffuseKernel(DiffuseParams params);

