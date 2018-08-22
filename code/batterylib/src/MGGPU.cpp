

#include "MGGPU.h"

#include <iostream>

#include "cuda/MGGPU.cuh"
#include "CudaUtility.h"
#include <chrono>



using namespace blib;

template class MGGPU<double>;

MGGPU_Volume toMGGPUVolume(const VolumeChannel & volchan, int id = -1) {
	MGGPU_Volume mgvol;

	mgvol.surf = volchan.getCurrentPtr().getSurface();
	mgvol.res = make_uint3(volchan.dim().x, volchan.dim().y, volchan.dim().z);
	mgvol.type = volchan.type();
	mgvol.volID = id;

	return mgvol;
}


template <typename T>
MGGPU<T>::MGGPU()
{

}


template <typename T>
bool MGGPU<T>::prepare(const VolumeChannel & mask, Params params, Volume & volume){

	
	_params = params;
	_mask = &mask;
	_volume = &volume;


	alloc();

	

	//Generate continous domain at first level
	MGGPU_Volume MGmask = toMGGPUVolume(mask, 0);
	MGGPU_GenerateDomain(MGmask, _params.d0, _params.d1, _levels[0].domain);	

	//Restrict domain to further levels
	MGGPU_KernelPtr domainRestrKernel = (double *)MGGPU_GetDomainRestrictionKernel().v;
	for (int i = 1; i < _levels.size(); i++) {
		MGGPU_Convolve(_levels[i-1].domain, domainRestrKernel, 2, _levels[i].domain);
	}


	//Send system params to gpu
	MGGPU_SysParams sysp;
	sysp.highConc = double(1.0);
	sysp.lowConc = double(0.0);
	sysp.concetrationBegin = (getDirSgn(params.dir) == 1) ? sysp.highConc : sysp.lowConc;
	sysp.concetrationEnd = (getDirSgn(params.dir) == 1) ? sysp.lowConc : sysp.highConc;
	
	sysp.cellDim[0] = params.cellDim.x;
	sysp.cellDim[1] = params.cellDim.y;
	sysp.cellDim[2] = params.cellDim.z;
	sysp.faceArea[0] = sysp.cellDim[1] * sysp.cellDim[2];
	sysp.faceArea[1] = sysp.cellDim[0] * sysp.cellDim[2];
	sysp.faceArea[2] = sysp.cellDim[0] * sysp.cellDim[1];	

	sysp.dirPrimary = getDirIndex(params.dir);
	sysp.dirSecondary = make_uint2((sysp.dirPrimary + 1) % 3, (sysp.dirPrimary + 2) % 3);

	sysp.dir = params.dir;

	if (!commitSysParams(sysp)) {
		return false;
	}	

	auto & sysTop = _levels[0].A;	
	sysTop.allocDevice(_levels[0].N(), sizeof(MGGPU_SystemTopKernel));
	
	
	
	MGGPU_GenerateSystemTopKernel(_levels[0].domain, (MGGPU_SystemTopKernel*)sysTop.gpu, _levels[0].f);
	
	{
		std::cout << "i " << 0 << ", kn: " << "N/A" << ", per row:" << 7 << ", n: " << _levels[0].N() << ", row: " << sizeof(MGGPU_SystemTopKernel) << "B" << ", total: " << (_levels[0].N() * sizeof(MGGPU_SystemTopKernel)) / (1024.0f * 1024.0f) << "MB" << std::endl;
	}
	
//debug
#ifdef DEBUG
	{		
		sysTop.retrieve();		
		auto & ftex = volume.getChannel(_levels[0].f.volID).getCurrentPtr();
		ftex.allocCPU();
		ftex.retrieve();


		char b;
		b = 0;
	}
#endif


	int prevKernelSize = 3;
	for (int i = 1; i < numLevels(); i++) {
		int kn = prevKernelSize;
		
		kn = std::max({ kn, INTERP_SIZE, kn + INTERP_SIZE - 1 });
		kn = std::max({ kn, RESTR_SIZE, kn + RESTR_SIZE - 1 });		


		//Kernel size
		//assert(kn == 3 + 5 * i);
		kn = 3 + 4 * i;
		//int kn = 3 + 5 * i;
		size_t rowsize = kn*kn*kn * sizeof(double);
		std::cout << "i " << i << ", kn: " << kn << ", per row:" << kn*kn*kn <<  ", n: " << _levels[i].N() << ", row: " << rowsize << "B" << ", total: " << (_levels[i].N() * rowsize) / (1024.0f * 1024.0f) << "MB" << std::endl;
		_levels[i].A.allocDevice(_levels[i].N(), rowsize);

		prevKernelSize = kn;
	}
	

	
	//Levle one mid
	/*MGGPU_RestrictKernel restrKernelMid  = MGGPU_GetRestrictionKernel(
		make_uint3(mask.dim().x / 4, mask.dim().y / 4, mask.dim().z / 4),
		make_uint3(mask.dim().x / 2, mask.dim().y / 2, mask.dim().z / 2) ,
		getDirIndex(sysp.dir)
	);	

	MGGPU_RestrictKernel restrKernelX0 = MGGPU_GetRestrictionKernel(
		make_uint3(mask.dim().x / 4, 0, mask.dim().z / 4),
		make_uint3(mask.dim().x / 2, mask.dim().y / 2, mask.dim().z / 2),
		getDirIndex(sysp.dir)
	);*/

	/*MGGPU_RestrictKernel restrKernelX0 = MGGPU_GetRestrictionKernel(
		make_uint3(mask.dim().x / 4, 0, mask.dim().z / 4),
		make_uint3(mask.dim().x / 2, mask.dim().y / 2, mask.dim().z / 2),
		getDirIndex(sysp.dir)
	);*/

	/*MGGPU_RestrictKernel restrKernelX1 = MGGPU_GetRestrictionKernel(
		make_uint3(1, mask.dim().y / 2, mask.dim().z / 2),
		make_uint3(mask.dim().x, mask.dim().y, mask.dim().z),
		getDirIndex(sysp.dir)
	);*/


	cudaDeviceSynchronize();

	cudaPrintMemInfo(0);

	return true;


	
	


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

	auto label = [](const std::string & prefix, int i) {
		char buf[16];
		itoa(i, buf, 10);
		return prefix + std::string(buf);
	};

	//Domain & vector allocation
	for (auto i = 0; i < numLevels(); i++) {
		int domain = _volume->addChannel(_levels[i].dim, TYPE_DOUBLE, false, label("domain", i));
		int x = _volume->addChannel(_levels[i].dim, TYPE_DOUBLE, false, label("x", i));
		int tmpx = _volume->addChannel(_levels[i].dim, TYPE_DOUBLE, false, label("tmpx", i));
		int r = _volume->addChannel(_levels[i].dim, TYPE_DOUBLE, false, label("r", i));
		int f = _volume->addChannel(_levels[i].dim, TYPE_DOUBLE, false, label("f", i));
		
		_levels[i].domain = toMGGPUVolume(_volume->getChannel(domain), domain);
		_levels[i].x = toMGGPUVolume(_volume->getChannel(x), x);
		_levels[i].tmpx = toMGGPUVolume(_volume->getChannel(tmpx), tmpx);
		_levels[i].r = toMGGPUVolume(_volume->getChannel(r), r);
		_levels[i].f = toMGGPUVolume(_volume->getChannel(f), f);
	}



	
	/*const auto origDim = _mask->dim();
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


		/ *levelMem[i] += Aval + Arowptr + Acolptr + b + x + tmpx + D;
		levelMem[i] += Rval + Rrowptr + Rcolptr;
		levelMem[i] += Ival + Irowptr + Icolptr;* /

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
*/


	

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


