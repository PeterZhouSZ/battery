#pragma once


#include <cuda_runtime.h>


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define _CUDA(x) cudaCheck(x, TOSTRING(x), __FILE__, __LINE__)

namespace blib {

	bool cudaCheck(
		cudaError_t result,
		const char * function,
		const char * file,
		int line,
		bool abort = true);
	
}
