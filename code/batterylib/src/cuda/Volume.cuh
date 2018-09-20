#include <cuda_runtime.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>

#include "CudaMath.h"
#include "PrimitiveTypes.h"

#include "VolumeTypes.cuh"


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

enum ReduceOpType {
	REDUCE_OP_SUM,
	REDUCE_OP_PROD,
	REDUCE_OP_SQUARESUM,
	REDUCE_OP_MIN,
	REDUCE_OP_MAX
};

#define VOLUME_REDUCTION_BLOCKSIZE 512
void launchReduceKernel(
	PrimitiveType type,
	ReduceOpType opType,
	uint3 res,
	cudaSurfaceObject_t surf,
	void * auxBufferGPU,
	void * auxBufferCPU,
	void * result
);



void launchClearKernel(
	PrimitiveType type, cudaSurfaceObject_t surf, uint3 res, void * val
);

void launchNormalizeKernel(
	PrimitiveType type, cudaSurfaceObject_t surf, uint3 res, double low, double high
);

/////////////////////////////////////////////

//Copies A to B
void launchCopyKernel(PrimitiveType type, uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B);

//C = A*B element wise
void launchMultiplyKernel(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t A,
	cudaSurfaceObject_t B,
	cudaSurfaceObject_t C
);

void launchDotProductKernel(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t A,
	cudaSurfaceObject_t B,
	cudaSurfaceObject_t C, //holds temporary product, TODO: direct reduction
	void * auxBufferGPU,
	void * auxBufferCPU,
	void * result
);

//C = A + beta * B
void launchAddAPlusBetaB(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t A,
	cudaSurfaceObject_t B,
	cudaSurfaceObject_t C,
	double beta
);

//A = gamma * (A + beta * B) + C
void launchAPlusBetaBGammaPlusC(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t A,
	cudaSurfaceObject_t B,
	cudaSurfaceObject_t C,
	double beta,
	double gamma
);

//A = A + beta * B + gamma * C
void launchABC_BetaGamma(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t A,
	cudaSurfaceObject_t B,
	cudaSurfaceObject_t C,
	double beta,
	double gamma
);