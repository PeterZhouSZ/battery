#include <cuda_runtime.h>

#include "CudaMath.h"
#include "PrimitiveTypes.h"


#define VOLUME_VOX					\
uint3 vox = make_uint3(			\
		blockIdx.x * blockDim.x,	\
		blockIdx.y * blockDim.y,	\
		blockIdx.z * blockDim.z		\
	) + threadIdx;					\


#define VOLUME_VOX_GUARD(res)					\
	VOLUME_VOX									\
	if (vox.x >= res.x || vox.y >= res.y || vox.z >= res.z)	\
	return;		

__host__ __device__ inline uint roundDiv(uint a, uint b) {
	return (a + (b - 1)) / b;
}

__host__ __device__ inline uint3 roundDiv(uint3 a, uint3 b) {
	return make_uint3(
		(a.x + (b.x - 1)) / b.x,
		(a.y + (b.y - 1)) / b.y,
		(a.z + (b.z - 1)) / b.z
	);
}

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


void launchErodeKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut);

void launchHeatKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut);

void launchBinarizeKernel(uint3 res, 
	cudaSurfaceObject_t surfInOut, 
	PrimitiveType type,
	float threshold //between 0 to 1
);

#define BOUNDARY_ZERO_GRADIENT 1e37f
struct DiffuseParams {
	uint3 res;
	float voxelSize; 
	cudaSurfaceObject_t mask;
	cudaSurfaceObject_t concetrationIn;
	cudaSurfaceObject_t concetrationOut;
	float zeroDiff;
	float oneDiff;
		
	float boundaryValues[6];
};

void launchDiffuseKernel(DiffuseParams params);



//Subtracts B from A (result in A) ... A = A -B
void launchSubtractKernel(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B);

//Inplace reduce sum
float launchReduceSumKernel(uint3 res, cudaSurfaceObject_t surf);


float launchReduceSumSlice(uint3 res, cudaSurfaceObject_t surf, Dir dir, void * output);

