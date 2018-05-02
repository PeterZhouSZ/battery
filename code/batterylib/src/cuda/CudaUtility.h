#pragma once


#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>

#include "../../include/BatteryLibDef.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define _CUDA(x) cudaCheck(x, TOSTRING(x), __FILE__, __LINE__)
#define _CUSOLVER(x) cusolverCheck(x, TOSTRING(x), __FILE__, __LINE__)
#define _CUSPARSE(x) cusparseCheck(x, TOSTRING(x), __FILE__, __LINE__)

namespace blib {

	bool cudaCheck(
		cudaError_t result,
		const char * function,
		const char * file,
		int line,
		bool abort = true);

	bool cusolverCheck(
		cusolverStatus_t result,
		const char * function,
		const char * file,
		int line,
		bool abort = true);

	bool cusparseCheck(
		cusparseStatus_t result,
		const char * function,
		const char * file,
		int line,
		bool abort = true);

	BLIB_EXPORT void cudaOccupiedMemory(size_t * total, size_t * occupied, int device = 0);

	BLIB_EXPORT void cudaPrintMemInfo(int device = 0);

	//todo performance:
	//https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
	
}
