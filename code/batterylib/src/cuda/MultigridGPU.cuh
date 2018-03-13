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
