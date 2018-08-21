

#include "MGGPU.h"

#include <iostream>

#include "cuda/MGGPU.cuh"


using namespace blib;

template class MGGPU<double>;




template <typename T>
MGGPU<T>::MGGPU()
{

}


template <typename T>
bool MGGPU<T>::prepare(const VolumeChannel & mask, Params params, VolumeChannel & debugChannel){

	_params = params;
	_mask = &mask;




	MGGPU_Volume MGmask;
	MGmask.surf = mask.getCurrentPtr().getSurface();
	MGmask.res = make_uint3(mask.dim().x, mask.dim().y, mask.dim().z);
	MGmask.type = mask.type();

	//VolumeChannel domain = VolumeChannel(mask.dim(), TYPE_DOUBLE, false);
	

	MGGPU_Volume MGDomain;
	MGDomain.surf = debugChannel.getCurrentPtr().getSurface();
	MGDomain.res = make_uint3(debugChannel.dim().x, debugChannel.dim().y, debugChannel.dim().z);
	MGDomain.type = debugChannel.type();

	MGGPU_GenerateDomain(MGmask, _params.d0, _params.d1, MGDomain);
	


	///*
	//
	//on the fly:
	//	I (weights (i+1), dir) -> convolution
	//	R (NO weights) -> convolution

	//*/

	///*
	//	Allocates memory for each level:
	//	Ab = x ...for equation
	//	tmpx ... for double buffering of x,
	//	r ... for residual,
	//	I,R .. for interp/restrict matrices
	//*/
	//alloc();

	///*
	//	Builds matrix A and vector b for the top level
	//*/
	//buildLinearSystem();

	///*
	//	Generate diffusion coefficients for lower levels by simple restriction
	//	Used to weigh interpolation matrix
	//	(Test if needed)
	//*/
	//subsampleDiffusion();

	///*		
	//	Generate R and I matrices, construct lower level A by I,A,R multiplication
	//*/
	//buildLevelsGalerkin();


	return false;

}


template <typename T>
bool MGGPU<T>::alloc()
{

	const auto origDim = _mask->dim();
	const size_t origTotal = origDim.x * origDim.y * origDim.z;

	
	_levels.resize(numLevels());
	_levels[0].dim = origDim;
	for (auto i = 1; i < numLevels(); i++) {
		//TODO: add odd size handling
		const auto prevDim = _levels[i - 1].dim;
		_levels[i].dim = { prevDim.x / 2, prevDim.y / 2, prevDim.z / 2 };
	}	

	//Calculate memory requirements
	const size_t perRowA = 7;
	const size_t perRowR = 4 * 4 * 4;
	const size_t perRowI = 8;

	size_t DBL = sizeof(double);

	std::vector<size_t> levelMem(numLevels(), 0);
	size_t totalMem = 0;
	for (auto i = 0; i < numLevels(); i++){
		size_t N = _levels[i].dim.x * _levels[i].dim.y * _levels[i].dim.z;
		size_t M = N;

		
		
		size_t Aval = N * perRowA * sizeof(double);
		size_t Arowptr = (M + 1) * sizeof(int); 
		size_t Acolptr = N * perRowA * sizeof(int);

		size_t D = N * sizeof(double);
		size_t b = N * sizeof(double);
		size_t x = N * sizeof(double);
		size_t tmpx = N * sizeof(double);

		size_t Rval = N * perRowR * sizeof(double);
		size_t Rrowptr = (M + 1) * sizeof(int);
		size_t Rcolptr = N * perRowR * sizeof(int);

		size_t Ival = N * perRowI * sizeof(double);
		size_t Irowptr = (M + 1) * sizeof(int);
		size_t Icolptr = N * perRowI * sizeof(int);


		/*levelMem[i] += Aval + Arowptr + Acolptr + b + x + tmpx + D;
		levelMem[i] += Rval + Rrowptr + Rcolptr;
		levelMem[i] += Ival + Irowptr + Icolptr;*/

		//A kernels
		int aidim = 3 + i * 4;
		levelMem[i] += N * DBL * aidim*aidim*aidim;
		//I kernels
		//levelMem[i] += N * DBL * 8;
		//levelMem[i] += N * DBL * 4; //x, tmpx, r, f
		//R ... on the fly

		
		totalMem += levelMem[i];
		std::cout << "Level " << i << " :" << float(levelMem[i]) / (1024.0f * 1024.0f) << "MB" << std::endl;
		std::cout << aidim << std::endl;
	}

	std::cout << "Total  " << float(totalMem) / (1024.0f * 1024.0f * 1024.0f) << "GB" << std::endl;


	

	return true;

}

template <typename T>
bool MGGPU<T>::buildLinearSystem()
{
	return false;
}

template <typename T>
bool MGGPU<T>::subsampleDiffusion()
{
	return false;
}




template <typename T>
bool MGGPU<T>::buildLevelsGalerkin()
{
	return false;
}


