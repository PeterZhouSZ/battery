#include "CudaUtility.h"

#include <iostream>


bool blib::cudaCheck(
	cudaError_t result, 
	const char * function, 
	const char * file,
	int line,
	bool abort)
{

	if (result == cudaSuccess) 
		return true;
	
	std::cerr	<< "CUDA Error: " << cudaGetErrorString(result) 
				<< "(" << function << " at " 
				<< file << ":" << line
				<< ")" 
				<< std::endl;
	
	if (abort)
		exit(result);

	return false;
}

