

#include "MGGPU.h"

#include <iostream>

#include "cuda/MGGPU.cuh"
#include "CudaUtility.h"
#include <chrono>
#include <numeric>


#include "JacobiSolver.h"
//#ifdef DEBUG
//#define SAVE_TO_FILE
//#endif

//#define SAVE_TO_FILE


#include <fstream>

#include "Timer.h"




using namespace blib;

template class MGGPU<double>;




template <size_t KN>
MGGPU<double>::DenseMat kernelToDenseMat(MGGPU_Kernel3D<KN> * kernels, ivec3 dim0, ivec3 dim1) {
	size_t rows = dim0.x * dim0.y * dim0.z;
	size_t cols = dim1.x * dim1.y * dim1.z;
	
	MGGPU<double>::DenseMat M = MGGPU<double>::DenseMat::Zero(rows, cols);

	ivec3 stride = { 1, dim1.x, dim1.x * dim1.y };

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




		for (int x = 0; x < KN; x++) {
			for (int y = 0; y < KN; y++) {
				for (int z = 0; z < KN; z++) {

					ivec3 offset = { x,y,z };
					if (KN % 2 == 0) {
						offset -= ivec3(KN / 2 - 1);
					}
					else {
						offset -= ivec3(KN / 2);
					}


					if (kernel.v[x][y][z] != 0.0) {
						ivec3 kpos = jpos + offset;
						if (isValidPos(dim1, kpos)) {
							size_t col = kpos.x * stride.x + kpos.y * stride.y + kpos.z * stride.z;
							M(i,col) = kernel.v[x][y][z];
						}
					}
				}
			}
		}

	}

	return M;



}

template <size_t KN>
MGGPU<double>::SparseMat kernelToSparse(MGGPU_Kernel3D<KN> * kernels, ivec3 dim0, ivec3 dim1) {

	size_t rows = dim0.x * dim0.y * dim0.z;
	size_t cols = dim1.x * dim1.y * dim1.z;

	Eigen::SparseMatrix<double, Eigen::RowMajor> M;

	M.resize(rows, cols);
	M.reserve(Eigen::VectorXi::Constant(rows, KN*KN*KN));

	ivec3 stride = { 1, dim1.x, dim1.x * dim1.y };

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
						offset -= ivec3(KN / 2 - 1);
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


		/*if (i < 32) {
			std::cout << i << " -> " << nonzeros << std::endl;
		}*/

	}

	return M;




}



MGGPU<double>::Vector volumeToVector(const MGGPU_Volume & vol){
	assert(vol.cpu);
	size_t N = vol.res.x * vol.res.y * vol.res.z;
	MGGPU<double>::Vector V = Eigen::Map< MGGPU<double>::Vector >((double*)vol.cpu, N);
	return V;
}

void vectorToVolume(const MGGPU<double>::Vector & v, const MGGPU_Volume & vol) {
	//assert(vol.cpu);
	size_t N = vol.res.x * vol.res.y * vol.res.z;
	memcpy(vol.cpu, v.data(), N * sizeof(double));
	
}




inline uint3 make_uint3(const ivec3 & i) {
	return make_uint3(i.x, i.y, i.z);
}



bool saveSparse(const MGGPU<double>::SparseMat & M, const std::string & name, int level) {
	
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
MGGPU<T>::MGGPU() :
	_iterations(0)
{

}


template <typename T>
bool blib::MGGPU<T>::saveVolume(MGGPU_Volume & v, const std::string & name, int level)
{
	auto & ptr = _volume->getChannel(v.volID).getCurrentPtr();
	ptr.retrieve();
	return saveVector<T>(v.cpu, v.res.x*v.res.y*v.res.z, name, level);
}

template <typename T>
void blib::MGGPU<T>::retrieveVolume(MGGPU_Volume & v)
{
	auto & ptr = _volume->getChannel(v.volID).getCurrentPtr();
	ptr.retrieve();
}

template <typename T>
void blib::MGGPU<T>::commitVolume(MGGPU_Volume & v)
{
	auto & ptr = _volume->getChannel(v.volID).getCurrentPtr();
	ptr.commit();
}



template <typename T>
bool MGGPU<T>::prepare(VolumeChannel & mask, PrepareParams params, Volume & volume) {



	_params = params;
	_mask = &mask;
	_volume = &volume;

	//cudaPrintProperties();

	alloc();

	


	/*
		Generate continous domain at first level
	*/
	{
		MGGPU_Volume MGmask = toMGGPUVolume(mask, 0);
		CUDATimer tGenDomain(true);
		MGGPU_GenerateDomain(MGmask, _params.d0, _params.d1, _levels[0].domain);
		std::cout << "Gen domain " << tGenDomain.time() << "s" << std::endl;

/*#ifdef SAVE_TO_FILE
		auto & D0ptr = volume.getChannel(_levels[0].domain.volID).getCurrentPtr();
		D0ptr.retrieve();
		saveVector<double>(D0ptr.getCPU(), _levels[0].N(), "MGGPU_D", 0);
#endif*/
	}

	

	/*
		Restrict domain to further levels
	*/
	{
		MGGPU_KernelPtr domainRestrKernel = (double *)MGGPU_GetDomainRestrictionKernel().v;
		CUDATimer tConvolveDomain(true);
		for (int i = 1; i < _levels.size(); i++) {
			MGGPU_Convolve(_levels[i - 1].domain, domainRestrKernel, 2, _levels[i].domain);

/*#ifdef SAVE_TO_FILE
			auto & ptr = volume.getChannel(_levels[i].domain.volID).getCurrentPtr();
			ptr.retrieve();
			saveVector<double>(ptr.getCPU(), _levels[i].N(), "MGGPU_D", i);
#endif*/
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
		std::cout << "Alloc A0" << std::endl;
		sysTop.allocDevice(_levels[0].N(), sizeof(MGGPU_SystemTopKernel));
		CUDATimer t(true);
		MGGPU_GenerateSystemTopKernel(_levels[0].domain, (MGGPU_SystemTopKernel*)sysTop.gpu, _levels[0].f);
		std::cout << "Gen A0: " << t.time() << "s" << std::endl;
	}

	

	/*
		Generate A1
	*/
	{

		DataPtr & I = _levels[0].I;
		std::cout << "Alloc I0" << std::endl;
		I.allocDevice(_levels[0].N(), sizeof(MGGPU_Kernel3D<3>));
		CUDATimer tI(true);
		MGGPU_GenerateSystemInterpKernels(
			make_uint3(_levels[0].dim.x, _levels[0].dim.y, _levels[0].dim.z),
			_levels[1].domain,
			(MGGPU_Kernel3D<3> *)I.gpu
		);
		std::cout << "Gen I0: " << tI.time() << "s" << std::endl;

#ifdef SAVE_TO_FILE
		{
			
			I.retrieve();
			MGGPU_InterpKernel * ptr = (MGGPU_InterpKernel*)I.cpu;
			SparseMat mat = kernelToSparse<INTERP_SIZE>(
				(MGGPU_InterpKernel*)ptr,
				_levels[0].dim,
				_levels[1].dim
				);
			saveSparse(mat, "MGGPU_I", 0);
		}
#endif	


		DataPtr & A0 = _levels[0].A;


		DataPtr & A1 = _levels[1].A;
		std::cout << "Alloc A1" << std::endl;
		A1.allocDevice(_levels[1].N(), sizeof(MGGPU_Kernel3D<5>));



		auto Nres = make_uint3(_levels[0].dim.x, _levels[0].dim.y, _levels[0].dim.z);

		bool CPUtest = false;

		if (CPUtest) {
			I.retrieve();
			A0.retrieve();
			A1.retrieve();

			auto t0 = std::chrono::system_clock::now();
			MGGPU_BuildA1(
				Nres,
				(MGGPU_SystemTopKernel*)A0.cpu,
				(MGGPU_InterpKernel*)I.cpu,
				(MGGPU_Kernel3D<5>*)A1.cpu,
				false
			);
			auto t1 = std::chrono::system_clock::now();
			std::chrono::duration<double> prepTime = t1 - t0;
			std::cout << "Build A1 CPU: " << prepTime.count() << "s" << std::endl;

			A1.commit();
		}

		{

			cudaPrintMemInfo();
			CUDATimer t(true);
			MGGPU_BuildA1(
				Nres,
				(MGGPU_SystemTopKernel*)A0.gpu,
				(MGGPU_InterpKernel*)I.gpu,
				(MGGPU_Kernel3D<5>*)A1.gpu,
				true
			);
			t.stop();
			_CUDA(cudaPeekAtLastError());
			std::cout << "Build A1 GPU: " << t.time() << "s" << std::endl;
			cudaPrintMemInfo();

		}


#ifdef SAVE_TO_FILE
		{
			DataPtr & A = _levels[1].A;

			A.retrieve();
			MGGPU_Kernel3D<5> * ptr = (MGGPU_Kernel3D<5> *)A.cpu;
			SparseMat mat = kernelToSparse<5>(
				(MGGPU_Kernel3D<5>*)ptr,
				_levels[1].dim,
				_levels[1].dim
				);
			saveSparse(mat, "MGGPU_A", 1);
		}
#endif

	}

	


	/*
	Generate Ai
	*/
	for (auto i = 2; i < _levels.size(); i++) {

		DataPtr & I = _levels[i - 1].I;
		DataPtr & Aprev = _levels[i - 1].A;
		DataPtr & Anext = _levels[i].A;



		std::cout << "Alloc I" << i << ", dim:" << _levels[i - 1].dim.x << "~^3 -> " << _levels[i].dim.x << "~^3" << std::endl;
		/*if (_levels[i].dim.x % 2 != 0 || _levels[i].dim.y % 2 != 0 || _levels[i].dim.z % 2 != 0) {
			std::cout << "!!!!!!!!!!!!!! Odd dimension not supported yet !!!!!!!!!!!!!!" << std::endl;
			return false;
		}*/
		I.allocDevice(_levels[i - 1].N(), sizeof(MGGPU_Kernel3D<3>));

		CUDATimer tI(true);
		MGGPU_GenerateSystemInterpKernels(
			make_uint3(_levels[i - 1].dim.x, _levels[i - 1].dim.y, _levels[i - 1].dim.z),
			_levels[i].domain,
			(MGGPU_Kernel3D<3> *)I.gpu
		);
		std::cout << "Gen I" << i << ": " << tI.time() << "s" << std::endl;
#ifdef SAVE_TO_FILE
		{
			//DataPtr & I = _levels[i-1].I;

			I.retrieve();
			MGGPU_InterpKernel * ptr = (MGGPU_InterpKernel*)I.cpu;
			SparseMat mat = kernelToSparse<INTERP_SIZE>(
				(MGGPU_InterpKernel*)ptr,
				_levels[i - 1].dim,
				_levels[i].dim
				);
			saveSparse(mat, "MGGPU_I", i - 1);
		}
#endif	



		std::cout << "Alloc A" << i << std::endl;
		Anext.allocDevice(_levels[i].N(), sizeof(MGGPU_Kernel3D<5>));

		auto Nres = make_uint3(_levels[i - 1].dim.x, _levels[i - 1].dim.y, _levels[i - 1].dim.z);


		bool CPUtest = false;

		if (CPUtest) {
			I.retrieve();
			Aprev.retrieve();
			Anext.retrieve();

			auto t0 = std::chrono::system_clock::now();
			MGGPU_BuildAi(
				Nres,
				(MGGPU_Kernel3D<5>*)Aprev.cpu,
				(MGGPU_InterpKernel*)I.cpu,
				(MGGPU_Kernel3D<5>*)Anext.cpu,
				false
			);
			auto t1 = std::chrono::system_clock::now();
			std::chrono::duration<double> prepTime = t1 - t0;
			std::cout << "Build A1 CPU: " << prepTime.count() << "s" << std::endl;

			Anext.commit();
		}
		else {

			cudaPrintMemInfo();
			CUDATimer t(true);
			MGGPU_BuildAi(
				Nres,
				(MGGPU_Kernel3D<5>*)Aprev.gpu,
				(MGGPU_InterpKernel*)I.gpu,
				(MGGPU_Kernel3D<5>*)Anext.gpu,
				true
			);
			t.stop();
			_CUDA(cudaPeekAtLastError());
			std::cout << "Build A" << i << " GPU: " << t.time() << "s" << std::endl;
			cudaPrintMemInfo();

		}

#ifdef SAVE_TO_FILE
		{
			DataPtr & A = _levels[i].A;

			A.retrieve();
			MGGPU_Kernel3D<5> * ptr = (MGGPU_Kernel3D<5> *)A.cpu;
			SparseMat mat = kernelToSparse<5>(
				(MGGPU_Kernel3D<5>*)ptr,
				_levels[i].dim,
				_levels[i].dim
				);
			saveSparse(mat, "MGGPU_A", i);
		}
#endif	

	}


	

	//Retrieve last A from gpu and convert it to sparse/dense matrix
	{

		auto t0 = std::chrono::system_clock::now();
		DataPtr & lastA = _levels.back().A;
		const ivec3 dim = _levels.back().dim;
		lastA.retrieve();

		//_lastLevelA = kernelToDenseMat((MGGPU_Kernel3D<5>*)lastA.cpu, dim, dim);
		_lastLevelA = kernelToSparse((MGGPU_Kernel3D<5>*)lastA.cpu, dim, dim);
		_lastLevelSolver.analyzePattern(_lastLevelA);
		_lastLevelSolver.factorize(_lastLevelA);

		auto t1 = std::chrono::system_clock::now();
		std::chrono::duration<double> prepTime = t1 - t0;
		std::cout << "Presolve ALast: " << prepTime.count() << "s" << std::endl;

		//saveSparse(_lastLevelA, "MGGPU_lastLevelA", 0);

	}



#ifdef MGGPU_CPU_TEST
	for (auto i = 1; i < numLevels(); i++) {

		DataPtr & A = _levels[i].A;
		const ivec3 dim = _levels[i].dim;
		A.retrieve();
		_levels[i].Acpu = kernelToSparse((MGGPU_Kernel3D<5>*)A.cpu, dim, dim);

	}
#endif


	return true;

#ifdef ____OLD
	{

		/*
			Generate Ai, using Galerkin Ri-1 * Ai-1 * Ii-1
		*/

		CUDATimer tGenI(true);
		for (auto level = 1; level < _levels.size(); level++) {





			const int kI = 3;
			const int kAprev = (level == 1) ? 3 : 5;
			const int kR = 4;
			//int kAI = (level == 1) ? 3 : MGGPU_outputKernelSize(kAprev, kI, 2);
			//int kA = (level == 1) ? 5 : MGGPU_outputKernelSize(kR, kAI, 2);
			const int kAI = MGGPU_outputKernelSize(kAprev, kI, 2);
			const int kA = MGGPU_outputKernelSize(kR, kAI, 2);

			/*if (level == 1)
				kAI = 3;*/


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

			auto Nres = make_uint3(_levels[level - 1].dim.x, _levels[level - 1].dim.y, _levels[level - 1].dim.z);
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
						kAI,
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
						kAI,
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
					(MGGPU_KernelPtr)AI.gpu,
					kAI
				);
			}



			std::cout << "A alloc & gen" << level << std::endl;

			A.allocDevice(_levels[level].N(), sizeof(double)*kA*kA*kA);

			//A.retrieve();
			//AI.retrieve();		

			MGGPU_CombineKernelsRestrict(
				Nhalfres,
				Nres,
				Nres,
				Nhalfres,
				(MGGPU_KernelPtr)AI.gpu,
				kAI,
				(MGGPU_KernelPtr)A.gpu,
				true
			);

			//A.commit();

			{
				DataPtr & A = _levels[level].A;
				A.retrieve();
				MGGPU_Kernel3D<5> * ptr = (MGGPU_Kernel3D<5> *)A.cpu;
				SparseMat mat = kernelToSparse<5>(
					(MGGPU_Kernel3D<5>*)ptr,
					_levels[level].dim,
					_levels[level].dim
					);
				saveSparse(mat, "MGGPU_A", level);
			}


			_CUDA(cudaPeekAtLastError());




		}




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
				4,
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
							k = MGGPU_GetRestrictionKernel(make_uint3(x, y, z), Nhalfres, sysp.dirPrimary);

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
			std::cout << "i " << i << ", kn: " << kn << ", per row:" << kn*kn*kn << ", n: " << _levels[i].N() << ", row: " << rowsize << "B" << ", total: " << (_levels[i].N() * rowsize) / (1024.0f * 1024.0f) << "MB" << std::endl;
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

#endif
}


template <typename T>
bool MGGPU<T>::alloc()
{

	const auto origDim = _mask->dim();
	const size_t origTotal = origDim.x * origDim.y * origDim.z;

	_levels.resize(numLevels());
	_levels[0].dim = origDim;
	std::cout << "Level 0: " << origDim.x << " X " << origDim.y << " X " << origDim.z << std::endl;
	for (auto i = 1; i < numLevels(); i++) {
		
		const auto prevDim = _levels[i - 1].dim;
		_levels[i].dim = { 
			(prevDim.x + 1) / 2, 
			(prevDim.y + 1) / 2,
			(prevDim.z + 1) / 2
		};

		std::cout << "Level "<< i <<": " << _levels[i].dim.x << " X " << _levels[i].dim.y << " X " << _levels[i].dim.z << std::endl;
	}

	auto label = [](const std::string & prefix, int i) {
		char buf[16];
		itoa(i, buf, 10);
		return prefix + std::string(buf);
	};

	//Domain & vector allocation
	//Automatically allocated on CPU as well.
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



	//Auxiliary buffer for reduction
	{
		const size_t maxN = _levels[0].N();
		const size_t reduceN = maxN / VOLUME_REDUCTION_BLOCKSIZE;		
		_auxReduceBuffer.allocDevice(reduceN, sizeof(double));
		_auxReduceBuffer.allocHost();
	}
	

	return true;

}


template <typename T>
void blib::MGGPU<T>::profile()
{


	//Profile Gauss seidel
	{
		double tolerance = 1e-6;
		int preN = 1;
		Profiler profiler;

		int i = 1;
		int kmax = 1;
		for (auto k = 0; k < kmax; k++) {
			auto t = CUDATimer(true);
			MGGPU_SmootherParams sp;
			sp.A = (MGGPU_KernelPtr)_levels[i].A.gpu;
			sp.isTopLevel = (i == 0);
			sp.f = _levels[i].f;
			sp.x = _levels[i].x;
			sp.tmpx = _levels[i].tmpx;
			sp.r = _levels[i].r;
			sp.dir = _params.dir;

			sp.res = make_uint3(_levels[i].dim);
			sp.tolerance = tolerance;
			sp.auxBufferGPU = _auxReduceBuffer.gpu;
			sp.auxBufferCPU = _auxReduceBuffer.cpu;
			sp.iter = preN;
			double er = MGGPU_GaussSeidel(sp);

			profiler.add("smooth", t.time());
		}
		profiler.stop();
		std::cout << profiler.summary();

		
	}

}

template <typename T>
void blib::MGGPU<T>::reset()
{

	for (auto lv = 0; lv < numLevels(); lv++) {
		MGGPU_SetToZero(_levels[lv].x);
		MGGPU_SetToZero(_levels[lv].r);
		MGGPU_SetToZero(_levels[lv].tmpx);
	}

	_iterations = 0;

}


template <typename T>
T MGGPU<T>::solve(const SolveParams & solveParams) {

	
	
	const int lastLevel = numLevels() - 1;
		
	
	const std::vector<int> cycle = genCycle(solveParams.cycleType);
		
	MGGPU_Residual_TopLevel(
		make_uint3(_levels[0].dim),
		(MGGPU_SystemTopKernel *)_levels[0].A.gpu,
		_levels[0].x,
		_levels[0].f,
		_levels[0].r
	);

	double f0SqNorm = MGGPU_SquareNorm(make_uint3(_levels[0].dim), _levels[0].f, _auxReduceBuffer.gpu, _auxReduceBuffer.cpu);
	double r0SqNorm = MGGPU_SquareNorm(make_uint3(_levels[0].dim), _levels[0].r, _auxReduceBuffer.gpu, _auxReduceBuffer.cpu);	
	double lastError = sqrt(r0SqNorm / f0SqNorm);


	auto err = [&](int i){
		double fiSqNorm = MGGPU_SquareNorm(make_uint3(_levels[i].dim), _levels[i].f, _auxReduceBuffer.gpu, _auxReduceBuffer.cpu);
		double riSqNorm = MGGPU_SquareNorm(make_uint3(_levels[i].dim), _levels[i].r, _auxReduceBuffer.gpu, _auxReduceBuffer.cpu);
		return sqrt(riSqNorm / fiSqNorm);
	};


	if (solveParams.verbose) {
		std::cout << "Starting solve" << "\n";
		std::cout << "Params:" << "\n";
		std::cout << "\t" << "Alpha:\t\t" << solveParams.alpha << "\n";
		std::cout << "\t" << "CycleType:\t\t" << solveParams.cycleType << "\n";
		std::cout << "\t" << "Pre/Post N:\t\t" << solveParams.preN << "/" << solveParams.postN << "\n";
		std::cout << "\t" << "Tolerance:\t\t" << solveParams.tolerance << "\n";
		std::cout << "\t" << "Max iter.:\t\t" << solveParams.maxIter << "\n";

		std::cout << "Cycle:" << "\n";
		for (auto k : cycle) {
			std::cout << k << ", ";
		}
		std::cout << "\n";



	
	}

	//const double alpha = 1.0;

	
	Profiler profiler;		
	
	
	//TODO: idea use just one tmpx,x,for  f ... with the size of the largest and reuse?

	for (auto k = 0; k < solveParams.maxIter; k++) {

		int prevI = -1;
		for (auto i : cycle) {

			//Last level
			if (i == numLevels() - 1) {

				
				auto t = Timer(true);
				
				retrieveVolume(_levels[i].f);
				//Convert F, solve, convert X
				Vector F = volumeToVector(_levels[i].f);
				Vector X = _lastLevelSolver.solve(F); //todo possible optimization: inplace x.cpu
				//std::cout << "LASTLEVEL NORM " << X << std::endl;			

				
				vectorToVolume(X, _levels[i].x);
				commitVolume(_levels[i].x);
				
				profiler.add("lastLevelExact", t.time());

				//Commit X back to GPU
				//_volume->getChannel(_levels[i].x.volID).getCurrentPtr().commit();				

			}
			//Restrict
			else if (i >= prevI) {

				//std::cout << "(" << i << ") Restrict" << std::endl;

				//Zero out x
				if (i > 0) {
					MGGPU_SetToZero(_levels[i].x);
				}


				/*saveVolume(_levels[i].x, "cmp/x_pre", i); //identical
				saveVolume(_levels[i].f, "cmp/f_pre", i); //identical
				saveVolume(_levels[i].r, "cmp/r_pre", i); //identical*/

				
				//std::cout << "Pre smooth " << err(i) << std::endl;
#ifdef MGGPU_CPU_TEST
				if(i == 0)
#endif
				{
					auto t = CUDATimer(true);
					MGGPU_SmootherParams sp;
					sp.A = (MGGPU_KernelPtr)_levels[i].A.gpu;
					sp.isTopLevel = (i == 0);
					sp.f = _levels[i].f;
					sp.x = _levels[i].x;
					sp.tmpx = _levels[i].tmpx;
					sp.r = _levels[i].r;
					sp.dir = _params.dir;

					sp.res = make_uint3(_levels[i].dim);
					sp.tolerance = solveParams.tolerance;
					sp.auxBufferGPU = _auxReduceBuffer.gpu;
					sp.auxBufferCPU = _auxReduceBuffer.cpu;
					sp.iter = solveParams.preN;
					sp.alpha = solveParams.alpha;
					
					


					//sp.iter = 2 << i;// preN * i;
					double er = MGGPU_GaussSeidel(sp);		
					//MGGPU_Jacobi(sp);

					


					profiler.add("smooth", t.time());
					//std::cout << "### (" << i << ") R " << er << std::endl;
				}
#ifdef MGGPU_CPU_TEST
				else {
					
					retrieveVolume(_levels[i].f);
					retrieveVolume(_levels[i].x);
					retrieveVolume(_levels[i].r);
					auto & F = volumeToVector(_levels[i].f);
					auto & X = volumeToVector(_levels[i].x);
					auto & R = volumeToVector(_levels[i].r);

					double er = solveGaussSeidel<double>(_levels[i].Acpu, F, X, R, _levels[i].dim, tolerance, preN);
					
					vectorToVolume(F, _levels[i].f);//not needed
					vectorToVolume(X, _levels[i].x);
					vectorToVolume(R, _levels[i].r);//not needed
					

					commitVolume(_levels[i].f);
					commitVolume(_levels[i].x);
					commitVolume(_levels[i].r);
					//std::cout << "### (" << i << ") R " << er << std::endl;
				
				}
#endif

				

				/*saveVolume(_levels[i].x, "cmp/x_post", i); //different, some same
				saveVolume(_levels[i].f, "cmp/f_post", i); //identical
				saveVolume(_levels[i].r, "cmp/r_post", i); //different, some same*/


				

				//std::cout << "Post smooth " << err(i) << std::endl;

				//restrict
				//R*x multiply 

				auto t = CUDATimer(true);
				MGGPU_Restrict(
					_levels[i].r,
					_levels[i + 1].f
				);
				profiler.add("restrict", t.time());

			}
			//Interpolate
			else {

				//std::cout << "(" << i << ") Interpolate" << std::endl;

				_levels[i].I.retrieve();
				DataPtr & tmp = _levels[i].I;

				//interpolate
				auto t = CUDATimer(true);
				MGGPU_InterpolateAndAdd(
					_levels[i + 1].x, //nhalf
					_levels[i].x, //n
					(MGGPU_InterpKernel *)_levels[i].I.gpu //n
				);
				profiler.add("interpolate", t.time());
						

				/*if (i == numLevels() - 2) {
					retrieveVolume(_levels[i].x);
					auto & X = volumeToVector(_levels[i].x);
					std::cout << X << std::endl;
					return 1;
				}*/


				//smooth
#ifdef MGGPU_CPU_TEST
				if (i == 0)
#endif
				{
					auto t = CUDATimer(true);
					MGGPU_SmootherParams sp;
					sp.A = (MGGPU_KernelPtr)_levels[i].A.gpu;
					sp.isTopLevel = (i == 0);
					sp.f = _levels[i].f;
					sp.x = _levels[i].x;
					sp.tmpx = _levels[i].tmpx;
					sp.r = _levels[i].r;

					sp.res = make_uint3(_levels[i].dim);
					sp.tolerance = solveParams.tolerance;
					sp.auxBufferGPU = _auxReduceBuffer.gpu;
					sp.auxBufferCPU = _auxReduceBuffer.cpu;
					sp.iter = solveParams.postN;
					sp.alpha = solveParams.alpha;

					//sp.iter = postN * i;
					
					//std::cout << "Restr iter: " << sp.iter << ", i:" << i << std::endl;

					double er = MGGPU_GaussSeidel(sp);
					//MGGPU_Jacobi(sp);
					profiler.add("smooth", t.time());

					//sp.iter = 2 << i;
					
				

					//std::cout << "### (" << i << ") I " << er << std::endl;
				}
#ifdef MGGPU_CPU_TEST
				else {

					retrieveVolume(_levels[i].f);
					retrieveVolume(_levels[i].x);
					retrieveVolume(_levels[i].r);
					auto & F = volumeToVector(_levels[i].f);
					auto & X = volumeToVector(_levels[i].x);
					auto & R = volumeToVector(_levels[i].r);

					double er = solveGaussSeidel<double>(_levels[i].Acpu, F, X, R, _levels[i].dim, tolerance, postN);

					vectorToVolume(F, _levels[i].f);//not needed
					vectorToVolume(X, _levels[i].x);
					vectorToVolume(R, _levels[i].r);//not needed

					commitVolume(_levels[i].f);
					commitVolume(_levels[i].x);
					commitVolume(_levels[i].r);
					//std::cout << "### (" << i << ") I " << er << std::endl;

				}
#endif


			}

			

			prevI = i;
		}

		
		_iterations++;

		MGGPU_Residual_TopLevel(
			make_uint3(_levels[0].dim),
			(MGGPU_SystemTopKernel *)_levels[0].A.gpu,
			_levels[0].x,
			_levels[0].f,
			_levels[0].r
		);


		auto tresidualEnd = CUDATimer(true);
		double r0SqNorm = MGGPU_SquareNorm(make_uint3(_levels[0].dim), _levels[0].r, _auxReduceBuffer.gpu, _auxReduceBuffer.cpu);
		double err = sqrt(r0SqNorm / f0SqNorm);
		profiler.add("residualEnd", tresidualEnd.time());
		
		if(solveParams.verbose)
		{
			std::cout << "k = " << k << " err: " << err << ", ratio: " << err / lastError << std::endl;
		}

		if (err > lastError) {
			return -1.0;			
		}

		lastError = err;

		if (err < solveParams.tolerance || isinf(err) || isnan(err)) {
			break;
			
		}

	}

	profiler.stop();
	if (solveParams.verbose) {
		std::cout << profiler.summary();
	}



	//cudaDeviceSynchronize();	
	return T(lastError);

	

}



template <typename T>
std::vector<int> MGGPU<T>::genCycle(CycleType ctype) const{ 

	std::vector<int> cycle;

	int levels = numLevels();
	ivec3 maxDim = _levels[0].dim;

	if (maxDim.x <= 64 || maxDim.y <= 64 || maxDim.z <= 64)
		ctype = V_CYCLE;


	if (ctype == V_CYCLE) {
		for (auto i = 0; i != levels; i++) {
			cycle.push_back(i);
		}
		for (auto i = levels - 2; i != -1; i--) {
			cycle.push_back(i);
		}
	}
	else if (ctype == W_CYCLE) {
		//auto midLevel = (levels - 1) - 2;
		auto midLevel = levels - 2;;
		for (auto i = 0; i != levels; i++) {
			cycle.push_back(i);
		}

		for (auto i = levels - 2; i != (midLevel - 1); i--) {
			cycle.push_back(i);
		}
		for (auto i = midLevel; i != levels; i++) {
			cycle.push_back(i);
		}

		for (auto i = levels - 2; i != -1; i--) {
			cycle.push_back(i);
		}
	}
	else if (ctype == V_CYCLE_SINGLE) {
		cycle = { 0, 1, 0 };
	}



	return cycle;
}


template <typename T>
T blib::MGGPU<T>::tortuosity()
{		
	
	//TODO on gpu

	const auto & topLevel = _levels[0];
	const auto dim = topLevel.dim;

	auto & xChannel = _volume->getChannel(topLevel.x.volID);
	auto & maskChannel = *_mask;

	xChannel.getCurrentPtr().retrieve();
	maskChannel.getCurrentPtr().retrieve();

	T * xData = (T*)xChannel.getCurrentPtr().getCPU();
	uchar * maskData = (uchar*)maskChannel.getCurrentPtr().getCPU();

	
	const int primary = getDirIndex(_params.dir);
	const int secondary[2] = { (primary + 1) % 3, (primary + 2) % 3 };

	int n = dim[secondary[0]] * dim[secondary[1]];
	int k = (getDirSgn(_params.dir) == -1) ? 0 : dim[primary] - 1;

	//Porosity
	std::vector<size_t> psums(dim[0], T(0));
	#pragma omp parallel for
	for (auto i = 0; i < dim[0]; i++) {
		size_t psum = 0;
		for (auto j = 0; j < dim[1]; j++) {
			for (auto k = 0; k < dim[2]; k++) {
				psum += (maskData[linearIndex(dim, {i,j,k})] == 0) ? 1 : 0;
			}
		}
		psums[i] = psum;
	}
	size_t sumD0 = std::accumulate(psums.begin(), psums.end(), T(0));

	T porosity = T(sumD0) / T(dim.x * dim.y * dim.z);


	std::vector<T> isums(dim[secondary[0]], T(0));
	#pragma omp parallel for
	for (auto i = 0; i < dim[secondary[0]]; i++) {
		T jsum = T(0);
		for (auto j = 0; j < dim[secondary[1]]; j++) {
			ivec3 pos;
			pos[primary] = k;
			pos[secondary[0]] = i;
			pos[secondary[1]] = j;

			if (maskData[linearIndex(dim, pos)] == 0)
				jsum += xData[linearIndex(dim, pos)];
		}
		isums[i] = jsum;
	}

	T sum = std::accumulate(isums.begin(), isums.end(), T(0));

	T dc = sum / n;
	T dx = 1.0f / (dim[primary] + 1);
	T tau = porosity / (dc * dim[primary] * 2);

	_porosity = porosity;

	return tau;

	/*
	Calculate average of low concetration plane where mask == 1
	*/
	
	
	

	/*
	const T * concData = _x[0].data();
	const uchar * cdata = (uchar *)mask.getCurrentPtr().getCPU();

	const int primaryDim = getDirIndex(dir);
	const int secondaryDims[2] = { (primaryDim + 1) % 3, (primaryDim + 2) % 3 };


	int n = dim[secondaryDims[0]] * dim[secondaryDims[1]];
	int k = (getDirSgn(dir) == -1) ? 0 : dim[primaryDim] - 1;
	*/

	/*
	Calculate average in low concetration plane
	*/

	/*
	bool zeroOutPart = true;
	std::vector<T> isums(dim[secondaryDims[0]], T(0));

	#pragma omp parallel for
	for (auto i = 0; i < dim[secondaryDims[0]]; i++) {
		T jsum = T(0);
		for (auto j = 0; j < dim[secondaryDims[1]]; j++) {
			ivec3 pos;
			pos[primaryDim] = k;
			pos[secondaryDims[0]] = i;
			pos[secondaryDims[1]] = j;

			if (zeroOutPart && cdata[linearIndex(dim, pos)] == 0)
				jsum += concData[linearIndex(dim, pos)];
		}
		isums[i] = jsum;
	}

	T sum = std::accumulate(isums.begin(), isums.end(), T(0));

	double dc = sum / n;
	double dx = 1.0f / (dim[primaryDim] + 1);
	double tau = _porosity / (dc * dim[primaryDim] * 2);
	*/
	

	

}