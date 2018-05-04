#include <cuda_runtime.h>

#include "CudaMath.h"
#include "PrimitiveTypes.h"


struct SparseMatGPU {
	void * data;
	void * row;
	void * col;	
	size_t NNZ;
};

struct LevelGPU {
	SparseMatGPU A;
	void * f;
	void * x;
	void * tmpx;
	void * r;
	uint3 dim;
};

#define MAX_LEVELS 16

struct MultigridGPUParams {
	LevelGPU levels[MAX_LEVELS] ;
	int numLevels;	
	PrimitiveType type;
};


cudaError_t _commitGPUParamsImpl(const MultigridGPUParams & mgHost);


void MGPrepareSystem(int level, cudaSurfaceObject_t mask, double d0, double d1, double3 cellDim, Dir dir);