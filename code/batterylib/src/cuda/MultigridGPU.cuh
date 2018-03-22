#include <cuda_runtime.h>

#include "CudaMath.h"
#include "PrimitiveTypes.h"

void launchConvertMaskKernel(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t surfIn,
	cudaSurfaceObject_t surfOut,
	double v0,
	double v1
);

void launchRestrictionKernel(
	PrimitiveType type,
	cudaSurfaceObject_t surfSrc,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest,
	double multiplier
);

void launchWeightedRestrictionKernel(
	PrimitiveType type,
	cudaSurfaceObject_t surfSrc,
	cudaSurfaceObject_t surfWeight,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest
);

void launchWeightedInterpolationKernel(
	PrimitiveType type,
	cudaSurfaceObject_t surfSrc,
	cudaSurfaceObject_t surfWeight,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest
);


struct LinSysParams {
	PrimitiveType type;
	uint3 res;
	float3 cellDim;
	Dir dir;
	uint dirPrimary;
	uint2 dirSecondary;		
	void * matrixData;
	cudaSurfaceObject_t surfX;
	cudaSurfaceObject_t surfF;
	cudaSurfaceObject_t surfD;
	cudaSurfaceObject_t surfR;
};

void launchPrepareSystemKernel(LinSysParams params);

void clearSurface(PrimitiveType type, cudaSurfaceObject_t surf, uint3 res, void * val);

//A = A + B
void surfaceAddition(PrimitiveType type, cudaSurfaceObject_t A, cudaSurfaceObject_t B, uint3 res);
//A = A - B
void surfaceSubtraction(PrimitiveType type, cudaSurfaceObject_t A, cudaSurfaceObject_t B, uint3 res);

//r = f - A*x
void residual(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t surfR,
	cudaSurfaceObject_t surfF,
	cudaSurfaceObject_t surfX,
	void * matrixData
);


struct SmootherParams {
	PrimitiveType type;
	void * matrixData;
	cudaSurfaceObject_t surfB;
	cudaSurfaceObject_t surfX;
	cudaSurfaceObject_t surfXtemp;
	cudaSurfaceObject_t surfR;
	uint3 res;
	void * errorOut;
	void * auxBufferGPU;
	void * auxBufferCPU;
	void * tolerance;
	size_t maxIter;
};


void solveGaussSeidel(
	const SmootherParams & params
);



void solveJacobi(
	const SmootherParams & params
);