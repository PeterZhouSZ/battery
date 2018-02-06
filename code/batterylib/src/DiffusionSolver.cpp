#include "DiffusionSolver.h"

#include "CudaUtility.h"

using namespace blib;


//#define DS_USEGPU
//#define DS_LINSYS_TO_FILE

#include <chrono>
#include <iostream>
#include <fstream>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>

DiffusionSolver::DiffusionSolver(bool verbose)
	: _verbose(verbose),
	 _maxElemPerRow(7) //diagonal and +-x|y|z
{

#ifdef DS_USEGPU

		//Handle to cusolver lib
		_CUSOLVER(cusolverSpCreate(&_handle));

		//Handle to cusparse
		_CUSPARSE(cusparseCreate(&_cusparseHandle));

		//Create new stream
		_CUDA(cudaStreamCreate(&_stream));
		_CUSPARSE(cusparseSetStream(_cusparseHandle, _stream));

		//Matrix description
		_CUSPARSE(cusparseCreateMatDescr(&_descrA));
		_CUSPARSE(cusparseSetMatType(_descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
		//todo matrix type - perhaps symmetrical?

		_CUSPARSE(cusparseSetMatIndexBase(_descrA, CUSPARSE_INDEX_BASE_ZERO));

#endif // DS_USEGPU


}

blib::DiffusionSolver::~DiffusionSolver()
{
#ifdef DS_USEGPU
		if (_handle) { _CUSOLVER(cusolverSpDestroy(_handle)); }
		if (_cusparseHandle) { _CUSPARSE(cusparseDestroy(_cusparseHandle)); }
		if (_stream) { _CUDA(cudaStreamDestroy(_stream)); }
		if (_descrA) { _CUSPARSE(cusparseDestroyMatDescr(_descrA)); }

		if (_deviceA)  _CUDA(cudaFree(_deviceA));
		if (_deviceRowPtr)  _CUDA(cudaFree(_deviceRowPtr));
		if (_deviceColInd)  _CUDA(cudaFree(_deviceColInd));

		if (_deviceB)  _CUDA(cudaFree(_deviceB));
		if (_deviceX)  _CUDA(cudaFree(_deviceX));
	
#endif // DS_USEGPU

}

bool blib::DiffusionSolver::solve(VolumeChannel & volChannel, ivec3 subdim)
{

	const auto & c = volChannel;
	auto dim = glm::min(subdim, c.dim);
	

	
	if(_verbose)
		std::cout << "DIM " << dim.x << ", " << dim.y << ", " << dim.z << " nnz:" << dim.x*dim.y*dim.z / (1024 * 1024.0f) << "M" << std::endl;



	const float highConc = 1.0f;
	const float lowConc = 0.0f;

	const float d0 = 1.0f;
	const float d1 = 0.0001f;



	//Matrix dimensions	
	size_t N = dim.x * dim.y * dim.z; //cols
	size_t M = N; //rows
	size_t nnz = _maxElemPerRow * M; //total elems
	

	//Reallocate if needed
	if (nnz > _nnz) {
		//Update dims
		_nnz = nnz;
		_M = M;
		_N = N;

		#ifdef DS_USEGPU
			if (_deviceA) _CUDA(cudaFree(_deviceA));
			_CUDA(cudaMalloc(&_deviceA, _nnz * sizeof(float)));

			if (_deviceB) _CUDA(cudaFree(_deviceB));
			_CUDA(cudaMalloc(&_deviceB, _M * sizeof(float)));

			//todo: map to other vol. channel
			if (_deviceX) _CUDA(cudaFree(_deviceX));
			_CUDA(cudaMalloc(&_deviceX, _N * sizeof(float)));

			if (_deviceRowPtr) _CUDA(cudaFree(_deviceRowPtr));
			_CUDA(cudaMalloc(&_deviceRowPtr, (_M + 1) * sizeof(int))); //CSR format

			if (_deviceColInd) _CUDA(cudaFree(_deviceColInd));
			_CUDA(cudaMalloc(&_deviceColInd, _nnz * sizeof(int)));
		#endif
	}

	


	//Copy data
	if (_verbose)
		std::cout << nnz << " elems, " << "vol: " << dim.x << " ^3" << std::endl;


	auto start0 = std::chrono::system_clock::now();
	
#ifdef DS_USEGPU
	std::vector<float> cpuData; cpuData.reserve(_nnz);
	std::vector<int> cpuRowPtr(M + 1, -1);
	std::vector<int> cpuColInd; cpuColInd.reserve(_nnz);
	std::vector<float> cpuB(M);	
#else
	std::vector<Eigen::Triplet<float>> triplets;
	Eigen::VectorXf b(M);
	b.setZero();
	Eigen::VectorXf x(N);
#endif
			

	volChannel.getCurrentPtr().retrieve();
	assert(volChannel.type == TYPE_UCHAR);
	uchar * D = (uchar *)volChannel.getCurrentPtr().getCPU();

	const auto linIndex = [dim](int x, int y, int z) -> size_t {
		return x + dim.x * y + dim.x * dim.y * z;
	};

	const auto sample = [&linIndex, dim, D, d0, d1](int x, int y, int z) {
		x = std::clamp(x, 0, dim.x - 1);
		y = std::clamp(y, 0, dim.y - 1);
		z = std::clamp(z, 0, dim.z - 1);		
		return (D[linIndex(x, y, z)] == 0) ? d0 : d1;
	};


	const vec3 h = { 1.0f / (dim.x + 1), 1.0f / (dim.y + 1), 1.0f / (dim.z + 1) };
	const vec3 invH = { 1.0f / h.x, 1.0f / h.y, 1.0f / h.z };
	const vec3 invH2 = { invH.x*invH.x,invH.y*invH.y,invH.z*invH.z };

	int curRowPtr = 0;
	for (auto z = 0; z < dim.z; z++) {
		for (auto y = 0; y < dim.y; y++) {
			for (auto x = 0; x < dim.x; x++) {
				auto i = linIndex(x, y, z);

				float bval = 0.0f;
				float vals[7] = { 0.0f,0.0f ,0.0f ,0.0f ,0.0f ,0.0f ,0.0f };
				int colId[7];
				colId[0] = i - dim.y*dim.x;
				colId[1] = i - dim.x;
				colId[2] = i - 1;

				colId[3] = i;

				colId[4] = i + 1;
				colId[5] = i + dim.x;
				colId[6] = i + dim.y*dim.x;



				auto Dcur = vec3(sample(x, y, z)) * invH2; // vec x vec product

				auto Dpos = (vec3(
					sample(x + 1, y, z) * invH2.x,
					sample(x, y + 1, z) * invH2.y,
					sample(x, y, z + 1) * invH2.z
				) + vec3(Dcur)) * 0.5f;
				auto Dneg = (vec3(
					sample(x - 1, y, z) * invH2.x,
					sample(x, y - 1, z) * invH2.y,
					sample(x, y, z - 1) * invH2.z
				) + vec3(Dcur)) * 0.5f;

				auto diagVal = -(Dpos.x + Dpos.y + Dpos.z + Dneg.x + Dneg.y + Dneg.z);

				//Von neumann cond (first order accurate)
				if (y == 0) diagVal += Dneg.y;
				if (z == 0) diagVal += Dneg.z;
				if (y == dim.y - 1) diagVal += Dpos.y;
				if (z == dim.z - 1) diagVal += Dpos.z;

				//Boundary conditions
				if (x == 0) {
					bval = 0.0f - sample(0, y, z) * highConc * invH2.x;
				}
				else if (x == dim.x - 1) {
					bval = 0.0f - sample(dim.x - 1, y, z) * lowConc * invH2.x;
				}


				if (z > 0) vals[0] = Dneg.z;
				if (y > 0) vals[1] = Dneg.y;
				if (x > 0) vals[2] = Dneg.x;

				vals[3] = diagVal;

				if (x < dim.x - 1) vals[4] = Dpos.x;
				if (y < dim.y - 1) vals[5] = Dpos.y;
				if (z < dim.z - 1) vals[6] = Dpos.z;

				
#ifdef DS_USEGPU
					cpuB[i] = bval;
#else
					b[i] = bval;
#endif					
		
				int inRow = 0;
				for (auto k = 0; k < 7; k++) {					
					if(vals[k] == 0.0f) continue;
#ifdef DS_USEGPU
					cpuData.push_back(vals[k]);
					cpuColInd.push_back(colId[k]);
#else
					triplets.push_back(Eigen::Triplet<float>(i, colId[k], vals[k]));
#endif
					
					inRow++;
				}

#ifdef DS_USEGPU
				cpuRowPtr[i] = curRowPtr;
#endif
				curRowPtr += inRow;
			}
		}
	}

#ifdef DS_USEGPU
	cpuRowPtr.back() = curRowPtr;
#else
	Eigen::SparseMatrix<float> A(M, N);
	A.setFromTriplets(triplets.begin(), triplets.end());
#endif
	

#ifndef DS_USEGPU
#ifdef DS_LINSYS_TO_FILE
	{
		std::ofstream f("A.txt");
		f << A.toDense();
		f.close();
	}
	{
		std::ofstream f("b.vec");
		f << b;
		f.close();
	}
#endif
#endif
	

	auto end0 = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds0 = end0 - start0;
	if (_verbose) {
		std::cout << "prep elapsed time: " << elapsed_seconds0.count() << "s\n";
	}


#ifdef DS_USEGPU

	int singularity;
	float tol = 1.0e-9f;

	_CUDA(cudaMemcpy(_deviceA, cpuData.data(), _nnz * sizeof(float), cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(_deviceRowPtr, cpuRowPtr.data(), (_M + 1) * sizeof(int), cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(_deviceColInd, cpuColInd.data(), _nnz * sizeof(int), cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(_deviceB, cpuB.data(), M * sizeof(float), cudaMemcpyHostToDevice));

	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);


		//Host version
		/*
		_CUSOLVER(cusolverSpScsrlsvluHost(
		_handle,
		_N,
		cpuData.size(),
		_descrA,
		cpuData.data(),
		cpuRowPtr.data(),
		cpuColInd.data(),
		cpuB.data(),
		tol,
		0,
		cpuX.data(),
		&singularity
		));*/


		cudaEventRecord(start, _stream);
		_CUSOLVER(cusolverSpScsrlsvqr(
			_handle,
			_N,
			_nnz,
			_descrA,
			_deviceA,
			_deviceRowPtr,
			_deviceColInd,
			_deviceB,
			tol,
			0,
			_deviceX,
			&singularity
		));
		cudaEventRecord(stop, _stream);
		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		if (_verbose) {
			std::cout << "solve GPU elapsed time: " << milliseconds / 1000.0f << "s\n";
		}
	}
	
	std::vector<float> resDeviceX(N);
	_CUDA(cudaMemcpy(resDeviceX.data(), _deviceX, _N * sizeof(float), cudaMemcpyDeviceToHost));

	if(_verbose){
		double sum = 0.0;
		for (auto & v : resDeviceX) {
			sum += v;
		}

		sum /= resDeviceX.size();
		std::cout << "DEVICE avg conc: " << sum << std::endl;
	}
#else
	auto start = std::chrono::system_clock::now();

	Eigen::BiCGSTAB<Eigen::SparseMatrix<float>> stab;
	//Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> stab;		
	//Eigen::SparseLU<Eigen::SparseMatrix<float>> stab;		
	//stab.analyzePattern(A);
	//stab.factorize(A);
	//x = stab.solve(b);
	//x = stab.compute(A).solve(b);

	x.setLinSpaced(0.0f, 1.0f);
	x = stab.compute(A).solveWithGuess(b, x);


	auto end = std::chrono::system_clock::now();

	if (_verbose) {
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "#iterations:     " << stab.iterations() << std::endl;
		std::cout << "estimated error: " << stab.error() << std::endl;
		std::cout << "solve host elapsed time: " << elapsed_seconds.count() << "s\n";				
		std::cout << "host avg conc: " << x.mean() << std::endl;

	}

#ifdef DS_LINSYS_TO_FILE
	{
		std::ofstream f("x.vec");
		f << x;
		f.close();
	}
#endif


#endif

	
	
	
	return true;
}

