

#include "MGGPU.h"

#include <iostream>

#include "cuda/MGGPU.cuh"
#include "CudaUtility.h"
#include <chrono>


#ifdef DEBUG
#define SAVE_TO_FILE
#endif

//#define SAVE_TO_FILE



#ifdef SAVE_TO_FILE
#include <Eigen/Eigen>
#include <fstream>
using SparseMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

#else
using SparseMat = char;
#endif


using namespace blib;

template class MGGPU<double>;


#ifndef SAVE_TO_FILE

bool saveSparse(const SparseMat & M, const std::string & name, int level) {
	return true;
}

template<typename T>
bool saveVector(void * vin, size_t N, const std::string & name, int level) {
	return true;
}

template <size_t KN>
SparseMat kernelToSparse(MGGPU_Kernel3D<KN> * kernels, ivec3 dim0, ivec3 dim1) {
	return 0;
}

#else

bool saveSparse(const SparseMat & M, const std::string & name, int level) {
	
	char buf[24]; itoa(level, buf, 10);
	std::ofstream f(name + "_" + std::string(buf) + ".dat");

	for (auto k = 0; k < M.rows(); k++) {
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(M, k); it; ++it) {
			auto  j = it.col();
			f << k + 1 << " " << j + 1 << " " << it.value() << "\n";
		}

		if (M.rows() < 100 || k % (M.rows() / 100))
			f.flush();
	}

	return true;
}

template<typename T>
bool saveVector(void * vin, size_t N,  const std::string & name, int level) {
	{

		T * v = (T *)vin;
		
		char buf[24]; itoa(level, buf, 10);
		std::ofstream f(name + "_" + std::string(buf) + ".txt");
		for (auto i = 0; i < N; i++) {
			f << v[i] << "\n";
			if (N < 100 || i % (N / 100))
				f.flush();
		}
	}

	return true;
}


template <size_t KN>
SparseMat kernelToSparse(MGGPU_Kernel3D<KN> * kernels, ivec3 dim0, ivec3 dim1) {

	size_t rows = dim0.x * dim0.y * dim0.z;
	size_t cols = dim1.x * dim1.y * dim1.z;

	Eigen::SparseMatrix<double, Eigen::RowMajor> M;

	M.resize(rows, cols);
	M.reserve(Eigen::VectorXi::Constant(rows, KN*KN*KN));

	ivec3 stride = {1, dim1.x, dim1.x * dim1.y};

	for (auto i = 0; i < rows; i++) {
		MGGPU_Kernel3D<KN> & kernel = kernels[i];

		ivec3 ipos = posFromLinear(dim0, i);
		ivec3 jpos = ipos;
		if (dim0 != dim1) {
			jpos = {
				int(jpos.x / (dim0.x / float(dim1.x))),
				int(jpos.y / (dim0.y / float(dim1.y))),
				int(jpos.z / (dim0.z / float(dim1.z)))
			};
		}


		int nonzeros = 0;

		for (int x = 0; x < KN; x++) {
			for (int y = 0; y < KN; y++) {
				for (int z = 0; z < KN; z++) {

					ivec3 offset = { x,y,z };
					if (KN % 2 == 0) {
						offset -= ivec3( KN / 2 - 1);
					}
					else {
						offset -= ivec3(KN / 2);
					}
					

					if (kernel.v[x][y][z] != 0.0) {
						ivec3 kpos = jpos + offset;
						if (isValidPos(dim1, kpos)) {
							size_t col = kpos.x * stride.x + kpos.y * stride.y + kpos.z * stride.z;
							M.insert(i, col) = kernel.v[x][y][z];
							nonzeros++;
						}
					}
				}
			}
		}


		if (i < 32) {
			std::cout << i <<" -> " << nonzeros << std::endl;
		}

	}

	return M;




}

#endif

MGGPU_Volume toMGGPUVolume(VolumeChannel & volchan, int id = -1) {
	MGGPU_Volume mgvol;

	mgvol.surf = volchan.getCurrentPtr().getSurface();
	mgvol.res = make_uint3(volchan.dim().x, volchan.dim().y, volchan.dim().z);
	mgvol.type = volchan.type();
	mgvol.volID = id;
	mgvol.cpu = volchan.getCurrentPtr().getCPU();

	return mgvol;
}

MGGPU_Volume toMGGPUVolume(const VolumeChannel & volchan, int id = -1) {
	MGGPU_Volume mgvol;

	mgvol.surf = volchan.getCurrentPtr().getSurface();
	mgvol.res = make_uint3(volchan.dim().x, volchan.dim().y, volchan.dim().z);
	mgvol.type = volchan.type();
	mgvol.volID = id;
	mgvol.cpu = nullptr;

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

		
	/*
		Generate continous domain at first level
	*/
	{		
		MGGPU_Volume MGmask = toMGGPUVolume(mask, 0);
		CUDATimer tGenDomain(true); 		
		MGGPU_GenerateDomain(MGmask, _params.d0, _params.d1, _levels[0].domain);		
		std::cout << "Gen domain " << tGenDomain.time() << "s" << std::endl;

#ifdef SAVE_TO_FILE
		auto & D0ptr = volume.getChannel(_levels[0].domain.volID).getCurrentPtr();
		D0ptr.retrieve();
		saveVector<double>(D0ptr.getCPU(), _levels[0].N(), "MGGPU_D", 0);
#endif
	}
	


	/*
		Restrict domain to further levels
	*/
	{
		MGGPU_KernelPtr domainRestrKernel = (double *)MGGPU_GetDomainRestrictionKernel().v;
		CUDATimer tConvolveDomain(true);
		for (int i = 1; i < _levels.size(); i++) {
			MGGPU_Convolve(_levels[i - 1].domain, domainRestrKernel, 2, _levels[i].domain);

#ifdef SAVE_TO_FILE
			auto & ptr = volume.getChannel(_levels[i].domain.volID).getCurrentPtr();
			ptr.retrieve();
			saveVector<double>(ptr.getCPU(), _levels[i].N(), "MGGPU_D", i);
#endif
		}

		std::cout << "Convolve domain " << tConvolveDomain.time() << "s" << std::endl;
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


	/*
		Generate A0 - top level kernels
	*/
	{
		auto & sysTop = _levels[0].A;
		sysTop.allocDevice(_levels[0].N(), sizeof(MGGPU_SystemTopKernel));
		MGGPU_GenerateSystemTopKernel(_levels[0].domain, (MGGPU_SystemTopKernel*)sysTop.gpu, _levels[0].f);		
	}

	/*
		Generate Ai, using Galerkin Ri-1 * Ai-1 * Ii-1
	*/

	CUDATimer tGenI(true);
	for (auto level = 1; level < _levels.size(); level++) {

		const int kI = 3;
		const int kAprev = (level == 1) ? 3 : 5;
		const int kA = 5;
		const int kR = 4;
		const int kAI = MGGPU_outputKernelSize(kAprev, kI, 2);


		DataPtr & A = _levels[level].A;
		DataPtr & Aprev = _levels[level - 1].A;

		std::cout << "Level " << level << std::endl;
		cudaPrintMemInfo();
		
		std::cout << "AI alloc & gen" << level << std::endl;
		//Free when out of scope
		DataPtr AI;		
		AI.allocDevice(_levels[level - 1].N(), sizeof(double)*kAI*kAI*kAI);
		AI.memsetDevice(0);

		cudaPrintMemInfo();

		auto Nres = make_uint3(_levels[level-1].dim.x, _levels[level - 1].dim.y, _levels[level - 1].dim.z);
		auto Nhalfres = make_uint3(_levels[level].dim.x, _levels[level].dim.y, _levels[level].dim.z);

		//A*I multiply
		if (level == 1) {

			CUDATimer tA(true);				
			
			if (false) {
				Aprev.retrieve();
				AI.retrieve();

				MGGPU_CombineKernelsTopLevel(
					Nres,
					Nres,
					Nhalfres,
					(MGGPU_KernelPtr)Aprev.cpu,
					nullptr,
					kI,
					(MGGPU_KernelPtr)AI.cpu,
					_levels[level].domain,
					false
				);

				AI.commit();
			}
			else {
				MGGPU_CombineKernelsTopLevel(
					Nres,
					Nres,
					Nhalfres,
					(MGGPU_KernelPtr)Aprev.gpu,
					nullptr,
					kI,
					(MGGPU_KernelPtr)AI.gpu,
					_levels[level].domain,
					true
				);

			}
			_CUDA(cudaPeekAtLastError());

			std::cout << "########## MGGPU_CombineKernelsTopLevel" << level << " time: " << tA.time() << std::endl;
			

			/*{
				AI.retrieve();
				MGGPU_Kernel3D<4> * ptr = (MGGPU_Kernel3D<4> *)AI.cpu;
				SparseMat mat = kernelToSparse<4>(
					(MGGPU_Kernel3D<4>*)ptr,
					_levels[0].dim,
					_levels[1].dim
					);
				saveSparse(mat, "MGGPU_AI", level - 1);
			}*/

		}
		else {
			std::cout << "I alloc & gen" << level << std::endl;
			DataPtr & I = _levels[level - 1].I;
			I.allocDevice(_levels[level - 1].N(), sizeof(MGGPU_Kernel3D<3>));
			//std::cout << "I^3: " << (_levels[level - 1].N() * sizeof(MGGPU_Kernel3D<3>)) / (1024.0f * 1024.0f) << "MB" << std::endl;
			//std::cout << "I^2: " << (_levels[level - 1].N() * sizeof(MGGPU_Kernel3D<2>)) / (1024.0f * 1024.0f) << "MB" << std::endl;

			//cudaPrintMemInfo();

			MGGPU_GenerateSystemInterpKernels(
				make_uint3(_levels[level - 1].dim.x, _levels[level - 1].dim.y, _levels[level - 1].dim.z),
				_levels[level].domain,
				(MGGPU_Kernel3D<3> *)I.gpu
			);


			MGGPU_CombineKernelsGeneric(
				Nres,
				Nres,
				Nres,
				Nhalfres,
				(MGGPU_KernelPtr)Aprev.gpu, 
				kAprev,
				(MGGPU_KernelPtr)I.gpu,
				kI,
				(MGGPU_KernelPtr)AI.gpu
			);
		}

		

		std::cout << "A alloc & gen" << level << std::endl;

		A.allocDevice(_levels[level].N(), sizeof(double)*kA*kA*kA);


		MGGPU_CombineKernelsRestrict(
			Nhalfres,
			Nres,
			Nres,
			Nhalfres,
			(MGGPU_KernelPtr)AI.gpu, //I0
			kAI,
			(MGGPU_KernelPtr)A.gpu
		);
		_CUDA(cudaPeekAtLastError());

		

		
	}
	std::cout << "Generate Ii " << tGenI.time() << "s" << std::endl;

	return true;





	{
				

		DataPtr I0Kernels;
		I0Kernels.allocDevice(_levels[0].N(), sizeof(MGGPU_Kernel3D<3>));
		I0Kernels.memsetDevice(0);

		MGGPU_GenerateSystemInterpKernels(
			make_uint3(_levels[0].dim.x, _levels[0].dim.y, _levels[0].dim.z),
			_levels[1].domain,
			(MGGPU_Kernel3D<3> *)I0Kernels.gpu
		);
		
		//I0Kernels.retrieve();


		DataPtr AI0;
		AI0.allocDevice(_levels[0].N(), sizeof(MGGPU_Kernel3D<4>));
		AI0.memsetDevice(0);
		//AI0.retrieve();

		auto Nres = make_uint3(_levels[0].dim.x, _levels[0].dim.y, _levels[0].dim.z);
		auto Nhalfres = make_uint3(_levels[1].dim.x, _levels[1].dim.y, _levels[1].dim.z);
		//AI0



		CUDATimer tAI0(true);
		/*MGGPU_CombineKernelsGeneric(
			Nres,
			Nres,
			Nres,
			Nhalfres,
			(MGGPU_KernelPtr)A0.gpu, //A0
			3,
			(MGGPU_KernelPtr)I0Kernels.gpu, //I0
			3,
			(MGGPU_KernelPtr)AI0.gpu
		);*/
		
		//A0.retrieve();
		//I0Kernels.retrieve();
		//AI0.retrieve();
		DataPtr sysTop; //////////////////////////////////////////////

		MGGPU_CombineKernelsTopLevel(
			Nres,			
			Nres,
			Nhalfres,
			(MGGPU_KernelPtr)sysTop.gpu, //A0			
			(MGGPU_KernelPtr)I0Kernels.gpu, //I0
			3,
			(MGGPU_KernelPtr)AI0.gpu,
			_levels[1].domain
		);
		_CUDA(cudaPeekAtLastError());
		//AI0.commit();

		tAI0.stop();
		std::cout << "GPU AI0 combine: " << tAI0.time() << std::endl;


		///////////

		DataPtr A1;
		A1.allocDevice(_levels[1].N(), sizeof(MGGPU_Kernel3D<5>));
		A1.memsetDevice(0);

		DataPtr R0;
		{
			CUDATimer tr0(true);
			R0.alloc(_levels[1].N(), sizeof(MGGPU_RestrictKernel));
			R0.memsetDevice(0);
			for (int z = 0; z < Nhalfres.z; z++) {
				for (int y = 0; y < Nhalfres.y; y++) {					
					for (int x = 0; x < Nhalfres.x; x++) {		
						size_t i = linearIndex(_levels[1].dim, { x,y,z }); // kernel in n/2
						MGGPU_RestrictKernel & k = ((MGGPU_RestrictKernel *)R0.cpu)[i];
						k = MGGPU_GetRestrictionKernel(make_uint3( x,y,z ), Nhalfres, sysp.dirPrimary);

					}
				}
			}
			R0.commit();		
			std::cout << "R0 cpu->gpu: " << tr0.time() << std::endl;
			
		}


		/*R0.retrieve();
		AI0.retrieve();
		A1.retrieve();*/


		CUDATimer tA1(true);
		//MGGPU_CombineKernelsGeneric(
		//AI0.retrieve();
		//A1.retrieve();
		MGGPU_CombineKernelsRestrict(
			Nhalfres,
			Nres,
			Nres,
			Nhalfres,			
			(MGGPU_KernelPtr)AI0.gpu, //I0
			4,
			(MGGPU_KernelPtr)A1.gpu			
		);
		
		_CUDA(cudaPeekAtLastError());
		//_CUDA(cudaDeviceSynchronize());
		//A1.commit();
		tA1.stop();

		std::cout << "GPU A1 combine: " << tA1.time() << std::endl;

		std::cout << "==== GPU A1: " << tAI0.time() + tA1.time() << std::endl;
		
		
#ifdef SAVE_TO_FILE
		{
			AI0.retrieve();
			MGGPU_Kernel3D<4> * ptr = (MGGPU_Kernel3D<4> *)AI0.cpu;
			SparseMat mat = kernelToSparse<4>(
				(MGGPU_Kernel3D<4>*)ptr,
				_levels[0].dim,
				_levels[1].dim
				);
			saveSparse(mat, "MGGPU_AI", 0);
		}

		{
			R0.retrieve();
			MGGPU_Kernel3D<4> * ptr = (MGGPU_Kernel3D<4> *)R0.cpu;
			SparseMat mat = kernelToSparse<4>(
				(MGGPU_Kernel3D<4>*)ptr,
				_levels[1].dim,
				_levels[0].dim
				);
			saveSparse(mat, "MGGPU_R", 0);
		}

		{
			A1.retrieve();
			MGGPU_Kernel3D<5> * ptr = (MGGPU_Kernel3D<5> *)A1.cpu;
			SparseMat mat = kernelToSparse<5>(
				(MGGPU_Kernel3D<5>*)ptr,
				_levels[1].dim,
				_levels[1].dim
				);
			saveSparse(mat, "MGGPU_A", 1);
		}

		
#endif


		//RA0
		/*MGGPU_CombineKernels(
			Nhalfres,
			Nres,
			Nres,
			Nres,
			nullptr, //R0
			4,
			nullptr, //A0
			3,
			nullptr//RA0
		);*/

		// R0 (AI0)
		/*MGGPU_CombineKernels(
			Nhalfres,
			Nres,
			Nres,
			Nhalfres,
			nullptr, //R0
			4,
			nullptr, //A0
			3,
			nullptr//RA0
		);*/

	}
	return true;

	std::cout << "I0 kern size " << (_levels[1].N() * sizeof(MGGPU_Kernel3D<4>)) / (1024.0f * 1024.0f) << "MB" << std::endl;

	
	DataPtr IT0Kernels;
	IT0Kernels.allocDevice(_levels[1].N(), sizeof(MGGPU_Kernel3D<4>));
	IT0Kernels.memsetDevice(0);

	CUDATimer tIT0Kern(true);
	MGGPU_GenerateTranposeInterpKernels(
		make_uint3(_levels[0].dim.x, _levels[0].dim.y, _levels[0].dim.z),
		make_uint3(_levels[1].dim.x, _levels[1].dim.y, _levels[1].dim.z),
		_levels[1].domain,
		(MGGPU_Kernel3D<4> *)IT0Kernels.gpu
	);
		
	std::cout << "GPU I' Time: " << tIT0Kern.time() << "s" << std::endl;


	
	auto & sysTop = _levels[0].A;	
	sysTop.allocDevice(_levels[0].N(), sizeof(MGGPU_SystemTopKernel));
	CUDATimer tSysTop(true);
	MGGPU_GenerateSystemTopKernel(_levels[0].domain, (MGGPU_SystemTopKernel*)sysTop.gpu, _levels[0].f);
	std::cout << "GPU A0 time: " << tSysTop.time() << "s" << std::endl;
	{
		std::cout << "i " << 0 << ", kn: " << "N/A" << ", per row:" << 7 << ", n: " << _levels[0].N() << ", row: " << sizeof(MGGPU_SystemTopKernel) << "B" << ", total: " << (_levels[0].N() * sizeof(MGGPU_SystemTopKernel)) / (1024.0f * 1024.0f) << "MB" << std::endl;
	}

	
	
	CUDATimer tAI0(true);
	auto & sys1 = _levels[1].A;
	sys1.allocDevice(_levels[0].N(), sizeof(MGGPU_Kernel3D<3>));

	MGGPU_GenerateAI0(
		make_uint3(_levels[0].dim.x, _levels[0].dim.y, _levels[0].dim.z),
		make_uint3(_levels[1].dim.x, _levels[1].dim.y, _levels[1].dim.z),
		(MGGPU_SystemTopKernel*)sysTop.gpu, 		
		(MGGPU_Kernel3D<4>*)IT0Kernels.gpu,
		(MGGPU_Kernel3D<3>*)sys1.gpu
	);
	tAI0.stop();

	std::cout << "GPU AI time: " << tAI0.time() << "s" << std::endl;

	/*DataPtr I0Kernels;
	I0Kernels.allocDevice(_levels[0].N(), sizeof(MGGPU_InterpKernel));
	MGGPU_GenerateSystemInterpKernels(make_uint3(_levels[0].dim.x, _levels[0].dim.y, _levels[0].dim.z), _levels[1].domain, (MGGPU_InterpKernel *) I0Kernels.gpu);
	*/


	
	
	
//debug
#ifdef DEBUG
	{		
/*		sysTop.retrieve();		
		auto & ftex = volume.getChannel(_levels[0].f.volID).getCurrentPtr();
		ftex.allocCPU();
		ftex.retrieve();


		char b;
		b = 0;*/
	}

	{
		/*I0Kernels.retrieve();
		MGGPU_InterpKernel * ikern = (MGGPU_InterpKernel *)I0Kernels.cpu;
		SparseMat I0 = kernelToSparse<3>(
			(MGGPU_Kernel3D<3>*)ikern, 
			_levels[0].dim, 
			_levels[1].dim
			);
		saveSparse(I0, "MGGPU_I", 0);

		char b;
		b = 0;*/
	}

	{
		/*IT0Kernels.retrieve();
		MGGPU_Kernel3D<4> * ikern = (MGGPU_Kernel3D<4> *)IT0Kernels.cpu;
		SparseMat IT0 = kernelToSparse<4>(
			(MGGPU_Kernel3D<4>*)ikern,
			_levels[1].dim,
			_levels[0].dim
		);
		saveSparse(IT0, "MGGPU_IT", 0);

		char b;
		b = 0;*/
	}

	{
		sys1.retrieve();

		
		
		SparseMat AI0 = kernelToSparse<3>(
			(MGGPU_Kernel3D<3>*)sys1.cpu,
			_levels[0].dim,
			_levels[1].dim
		);
		saveSparse(AI0, "MGGPU_AI", 0);
		

		char b;
		b = 0;
	}
#endif


	return true;

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
		
#ifdef DEBUG
		_volume->getChannel(domain).getCurrentPtr().allocCPU();
#endif
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


