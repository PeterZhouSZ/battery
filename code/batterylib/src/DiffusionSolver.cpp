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

#include <omp.h>

#include "PrimitiveTypes.h"

#include <stack>



template <typename T>
void solveJacobi(
	const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
	const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
	Eigen::Matrix<T, Eigen::Dynamic, 1> & x,
	float tolerance = 1e-4,
	size_t maxIter = 20000,
	bool verbose = false

){
	Eigen::Matrix<T, Eigen::Dynamic, 1> xprime(x.size());	
	Eigen::Matrix<T, Eigen::Dynamic, 1> res(x.size());

	maxIter = (maxIter / 2) * 2; //divisible by two


	float bsqnorm = b.squaredNorm();
	float tol2 = tolerance * tolerance * bsqnorm;

	for (auto i = 0; i < maxIter; ++i) {

		auto & curX = (i % 2 == 0) ? x : xprime;
		auto & nextX = (i % 2 == 0) ? xprime : x;

		res = b - A*curX;

		float err = res.squaredNorm();// / bsqnorm;

		//float err = res.mean();

		if (verbose && i % 128 == 0) {

			float tol_error = sqrt(err / bsqnorm);

			std::cout << "jacobi i: " << i << " err: " << err << ", tol_error: " << tol_error << std::endl;
		}

		if (err <= tol2) {
			

			if (verbose) {
				float tol_error = sqrt(err / bsqnorm);
				std::cout << "solved " << err << " <= " << tol2 << std::endl;
				std::cout << "tol_error" << tol_error << std::endl;
			}

			if (&x != &curX) {
				std::swap(x, curX);
			}				
			break;
		}
		jacobiStep(A, b, curX, nextX);

	}

}

template <typename T>
void jacobiStep(
	const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
	const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
	Eigen::Matrix<T, Eigen::Dynamic, 1> & x,
	Eigen::Matrix<T, Eigen::Dynamic, 1> & xnew
) {

	
	#pragma omp parallel for
	for (auto i = 0; i < A.rows(); i++) {		

		float sum = 0.0f;
		float diag = 0.0f;
		for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it) {
			auto  j = it.col();
			if (j == i) {				
				diag = it.value();				
				continue;
			}
			sum += it.value() * x[j];
		}

		if (diag == 0.0f) {
			char k;
			k = 0;
		}		
		xnew[i] = (b[i] - sum) / diag;
	}	
}


enum NeighType {
	NODE, 
	DIRICHLET,
	VON_NEUMANN
};

#define NO_NODE (size_t(0)-1)

struct Neigh {
	union {
		int index; //if node type
		float value; //if vonneumman -> add to diagonal, if dirichlet -> subtract from b
	};
	NeighType type;
};

struct Node {
	Node(ivec3 pos_) : pos(pos_) {}
	ivec3 pos;  //for debug
	Neigh neigh[6];
};

size_t linearIndex(const ivec3 & dim, int x, int y, int z) {
	return x + dim.x * y + dim.x * dim.y * z;
}
size_t linearIndex(const ivec3 & dim, const ivec3 & pos) {
	return pos.x + dim.x * pos.y + dim.x * dim.y * pos.z;
}


void buildNodeList(std::vector<Node> & nodeList, std::vector<size_t> & indices, const VolumeChannel & c, size_t nodeIndex_) {

	const auto & dim = c.dim();
	const size_t stride[3] = { 1, dim.x, dim.x*dim.y };
	const uchar * cdata = (uchar *)c.getCurrentPtr().getCPU();
	//todo add bound.cond here
	/*if (curPos.x < 0 || curPos.y < 0 || curPos.z < 0 ||
		curPos.x >= dim.x || curPos.y >= dim.y || curPos.z >= dim.z
		) return;*/

	
	
	

	std::stack<ivec3> stackNodePos;
	

	for (auto y = 0; y < dim.y; y++) {
		for (auto z = 0; z < dim.z; z++) {
			stackNodePos.push({ 0,y,z });
		}
	}
	while (!stackNodePos.empty()) {

		ivec3 nodePos = stackNodePos.top();		
		stackNodePos.pop();

		auto i = linearIndex(dim, nodePos);

	
		

		//Node at d1, skip
		if (indices[i] == NO_NODE) {
			if(cdata[i] != 0)
				continue;

			nodeList.push_back(Node(nodePos));
			indices[i] = nodeList.size() - 1;
		}


		assert(cdata[i] == 0);

		size_t nodeIndex = indices[i];
		

		auto * n = &nodeList[nodeIndex];
		

		//Visited
		//if (indices[i] != NO_NODE) return;

		const float conc[2] = { 0.0f,1.0f };
		const int dirichletDir = 0; //x dir

		const float highConc = 1.0f;
		const float lowConc = 0.0f;
		const float d0 = 0.001f;

		static const ivec3 dirVecs[6] = {
			ivec3(1,0,0),
			ivec3(-1,0,0),
			ivec3(0,1,0),
			ivec3(0,-1,0),
			ivec3(0,0,1),
			ivec3(0,0,-1)
		};

		for (auto k = 0; k < (3); k++) {
			for (auto dir = 0; dir < 2; dir++) {
				const int neighDir = 2 * k + dir;


				//Box domain conditions
				if ((dir == 0 && n->pos[k] == dim[k] - 1) ||
					(dir == 1 && n->pos[k] == 0)) {
					//Dirichlet  
					if (k == dirichletDir) {
						n->neigh[neighDir].type = DIRICHLET;
						n->neigh[neighDir].value = conc[dir]; //x1 or x0 conc
					}
					else {
						n->neigh[neighDir].type = VON_NEUMANN;
						n->neigh[neighDir].value = d0;
					}
				}
				//
				else {

					const ivec3 dirVec = dirVecs[2 * k + dir];
					const int thisStride = -(dir * 2 - 1) * int(stride[k]); // + or -, stride in k dim

					if (cdata[i + thisStride] == 0) {

						n->neigh[neighDir].type = NODE;

						int ni = linearIndex(dim, n->pos + dirVec);
						if (indices[ni] == NO_NODE) {
							nodeList.push_back(Node(n->pos + dirVec));
							indices[ni] = nodeList.size() - 1;
							n = &nodeList[nodeIndex]; //refresh pointer if realloc happened
							n->neigh[neighDir].index = indices[ni];							
							stackNodePos.push(n->pos + dirVec);
						}
						else {
							n->neigh[neighDir].index = indices[ni];
						}


					}
					else {
						n->neigh[neighDir].type = VON_NEUMANN;
						n->neigh[neighDir].value = d0;
					}
				}
			}

		}

	}

/*

	if (n.pos.x == 0) {
		n.neigh[X_NEG].type = DIRICHLET;
		n.neigh[X_NEG].value = highConc;
	}
	else {
		if (cdata[i - stride[0]] == 0){
			int ni = linearIndex(dim, n.pos + ivec3(-1, 0, 0));
			if (indices[ni] == NO_NODE) {
				nodeList.push_back(Node(n.pos + ivec3(-1, 0, 0)));
				indices[ni] = nodeList.size() - 1;
				buildNodeList(nodeList, indices, c, nodeList.back());
			}
			n.neigh[X_NEG].index = indices[ni];
			n.neigh[X_NEG].type = NODE;
		}
		else {
			n.neigh[X_NEG].type = VON_NEUMANN;
			n.neigh[X_NEG].value = d0;
		}
	}

	if (n.pos.x == dim.x - 1) {
		n.neigh[X_POS].type = DIRICHLET;
		n.neigh[X_POS].value = lowConc;
	}
	else {
		if (cdata[i + stride[0]] == 0) {
			int ni = linearIndex(dim, n.pos + ivec3(1, 0, 0));
			if (indices[ni] == NO_NODE) {
				nodeList.push_back(Node(n.pos + ivec3(1, 0, 0)));
				indices[ni] = nodeList.size() - 1;
				buildNodeList(nodeList, indices, c, nodeList.back());
			}
			n.neigh[X_POS].index = indices[ni];
			n.neigh[X_POS].type = NODE;
		}
		else {
			n.neigh[X_POS].type = VON_NEUMANN;
			n.neigh[X_POS].value = d0;
		}
	}
*/


/*

	if (n.pos.x == dim.x - 1) {
		n.neigh[X_NEG].type = DIRICHLET;
		n.neigh[X_NEG].value = lowConc;
	}
	else {
		

	}*/

	





}



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

bool blib::DiffusionSolver::solve(
	VolumeChannel & volChannel, 
	VolumeChannel * outVolume, 
	float d0,
	float d1,
	float tolerance
	
)
{

	const auto & c = volChannel;
	auto dim = c.dim();
	

	
	if(_verbose)
		std::cout << "DIM " << dim.x << ", " << dim.y << ", " << dim.z << " nnz:" << dim.x*dim.y*dim.z / (1024 * 1024.0f) << "M" << std::endl;



	const float highConc = 1.0f;
	const float lowConc = 0.0f;

	//const float d0 = 1.0f;
	//const float d1 = 0.0001f;



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
	Eigen::VectorXf X(N);	
#endif
			

	volChannel.getCurrentPtr().retrieve();
	assert(volChannel.type() == TYPE_UCHAR);
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

				assert(diagVal != 0.0f);

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

					//initial guess
					X[i] = 1.0f - (x / float(dim.x + 1));
					//X[i] = 0.0f;
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
	Eigen::SparseMatrix<float,Eigen::RowMajor> A(M, N);
	A.setFromTriplets(triplets.begin(), triplets.end());
	A.makeCompressed();
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

	omp_set_num_threads(8);
	Eigen::setNbThreads(8);


	//auto Xinit = X;

	//solveJacobi<float>(A, b, X, tolerance, 150000, true);
	

	//X = Xinit;
	
	Eigen::BiCGSTAB<Eigen::SparseMatrix<float,Eigen::RowMajor>> stab;
	//Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> stab;		
	//Eigen::SparseLU<Eigen::SparseMatrix<float>> stab;		
	//stab.analyzePattern(A);
	//stab.factorize(A);
	//x = stab.solve(b);
	//x = stab.compute(A).solve(b);

	const int maxIter = 3000;
	const int iterPerStep = 100;

	stab.setTolerance(tolerance);
	stab.setMaxIterations(iterPerStep);
	stab.compute(A);





	for (auto i = 0; i < maxIter; i += iterPerStep) {
		X = stab.solveWithGuess(b, X);
		float er = stab.error();
		if (_verbose) {
			std::cout << "i:" << i << ", estimated error: " << er << std::endl;
		}
		if (er <= tolerance)
			break;
	}	




	auto end = std::chrono::system_clock::now();

	if (_verbose) {
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "#iterations:     " << stab.iterations() << std::endl;
		std::cout << "estimated error: " << stab.error() << std::endl;
		std::cout << "solve host elapsed time: " << elapsed_seconds.count() << "s\n";				
		std::cout << "host avg conc: " << X.mean() << std::endl;
		std::cout << "tolerance: " << tolerance << std::endl;

		std::cout << "res.norm : " << (b - A*X).norm()  << std::endl;
		

	}

#ifdef DS_LINSYS_TO_FILE
	{
		std::ofstream f("x.vec");
		f << x;
		f.close();
	}
#endif


#endif


	if (outVolume) {
#ifdef DS_USEGPU
		memcpy(outVolume->getCurrentPtr().getCPU(), resDeviceX.data(), resDeviceX.size() * sizeof(float));
#else		

		memcpy(outVolume->getCurrentPtr().getCPU(), X.data(), X.size() * sizeof(float));	
#endif
	
	
	}
	
	
	
	return true;
}












BLIB_EXPORT bool blib::DiffusionSolver::solveWithoutParticles(
	VolumeChannel & volChannel, 
	VolumeChannel * outVolume, 
	float d0,
	float d1,
	float tolerance
)
{
#ifdef DS_USEGPU
	assert(false); // CPU only
#endif

	using T = float;

	const auto & c = volChannel;
	auto dim = c.dim();

	if (_verbose)
		std::cout << "DIM " << dim.x << ", " << dim.y << ", " << dim.z << " nnz:" << dim.x*dim.y*dim.z / (1024 * 1024.0f) << "M" << std::endl;


	const float highConc = 1.0f;
	const float lowConc = 0.0f;
	//const float d0 = 0.001f;

	//Max dimensions
	size_t maxN = dim.x * dim.y * dim.z; //cols
	size_t maxM = maxN; //rows	


	auto start0 = std::chrono::system_clock::now();

	std::vector<Node> nodeList;
	nodeList.reserve(maxN); 
	std::vector<size_t> indices(volChannel.dim().x*volChannel.dim().y*volChannel.dim().z, 0-1);
	buildNodeList(nodeList, indices, volChannel, 1);


	float porosity = 0.0f;
	{
		const uchar * cdata = (uchar *)volChannel.getCurrentPtr().getCPU();
		size_t cnt = 0;
		for (auto i = 0; i < volChannel.getCurrentPtr().byteSize() / sizeof(uchar); i++) {
			if (cdata[i] == 0)
				cnt++;
		}
		porosity = (cnt / float(maxN));
	}
	
	if (_verbose)
	{		
		std::cout << "occupancy (d0, direct): " << 100.0f * porosity << std::endl;
	}

	if (_verbose)
		std::cout << "occupancy (d0): " << 100.0f * nodeList.size() / float(dim.x*dim.y*dim.z) << "%  (" << nodeList.size() / (1024 * 1024.0f) << "M)" << std::endl;

	auto end0= std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds0 = end0 - start0;
	if (_verbose) {
		std::cout << "prep elapsed time: " << elapsed_seconds0.count() << "s\n";
	}



	////////////////////////////

	size_t M = nodeList.size();
	size_t N = M;

	const vec3 h = { 1.0f / (dim.x + 1), 1.0f / (dim.y + 1), 1.0f / (dim.z + 1) };
	const vec3 invH = { 1.0f / h.x, 1.0f / h.y, 1.0f / h.z };
	const vec3 invH2 = { invH.x*invH.x,invH.y*invH.y,invH.z*invH.z };


	std::vector<Eigen::Triplet<T>> triplets;
	
	Eigen::Matrix<T, Eigen::Dynamic, 1> b(M);
	b.setZero();
	Eigen::Matrix<T, Eigen::Dynamic, 1> x(N);

	size_t row = 0;
	for (auto & n : nodeList) {

		auto Dcur = vec3(d0) * invH2; // vec x vec product
		auto Dpos = Dcur;
		auto Dneg = Dcur;

		auto diagVal = -(Dpos.x + Dpos.y + Dpos.z + Dneg.x + Dneg.y + Dneg.z);
		float bval = 0.0f;

		int k = 0;
		for (auto & neigh : n.neigh) {
			float dval = (k % 2 == 0) ? Dpos[k / 2] : Dneg[k / 2];

			if (neigh.type == NODE) {
				triplets.push_back(Eigen::Triplet<T>(row, neigh.index, 
					dval
					));
			}
			else if (neigh.type == DIRICHLET) {
				bval -= d0 * neigh.value * invH2[k / 3];
			}
			else if (neigh.type == VON_NEUMANN) {
				diagVal += dval;
			}
			k++;
		}

		b[row] = bval;

		//initial guess
		x[row] = 1.0f - (n.pos.x / float(dim.x + 1));
		
		triplets.push_back(Eigen::Triplet<T>(row, row, diagVal));
		
		row++;
	}

	auto start = std::chrono::system_clock::now();
	Eigen::SparseMatrix<T, Eigen::RowMajor> A(M, N);
	A.setFromTriplets(triplets.begin(), triplets.end());
	A.makeCompressed();


	omp_set_num_threads(8);
	Eigen::setNbThreads(8);

	/*Eigen::VectorXf res;
	for (auto i = 0; i < 1024; i++) {
		jacobi(A, b, x);
		res = A*b - x;
		std::cout << "jacobi res: " << res.mean() << std::endl;
	}*/

	Eigen::BiCGSTAB<Eigen::SparseMatrix<T, Eigen::RowMajor>> stab;	

	
	stab.setTolerance(tolerance);
	stab.compute(A);
	

	const int maxIter = 3000;
	const int iterPerStep = 100;

	stab.setMaxIterations(iterPerStep);

	for (auto i = 0; i < maxIter; i += iterPerStep) {
		x = stab.solveWithGuess(b, x);		
		float er = stab.error();		
		if (_verbose) {			
			std::cout << "i:" << i << ", estimated error: " << er << std::endl;
		}
		if (er <= tolerance)
			break;
	}	
	
	auto end = std::chrono::system_clock::now();


	if (_verbose) {
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "#iterations:     " << stab.iterations() << std::endl;
		std::cout << "estimated error: " << stab.error() << std::endl;
		std::cout << "solve host elapsed time: " << elapsed_seconds.count() << "s\n";
		std::cout << "host avg conc: " << x.mean() << std::endl;
		std::cout << "tolerance " << tolerance << std::endl;

	}

	//Convert solution to volume
	{
		float * concData = (float *)outVolume->getCurrentPtr().getCPU();
		const uchar * cdata = (uchar *)volChannel.getCurrentPtr().getCPU();
		outVolume->clearCurrent();
		int nodeIndex = 0;
		for (auto & n : nodeList) {
			auto i = linearIndex(dim, n.pos);
			concData[i] = x[nodeIndex];			
			nodeIndex++;

		}
	}

	return true;
}

BLIB_EXPORT float blib::DiffusionSolver::tortuosityCPU(const VolumeChannel & mask, const VolumeChannel & concetration, Dir dir)
{
	
	
	assert(mask.type() == TYPE_UCHAR || mask.type() == TYPE_CHAR);
	assert(concetration.type() == TYPE_FLOAT);
	assert(mask.dim().x == concetration.dim().x);
	assert(mask.dim().y == concetration.dim().y);
	assert(mask.dim().z == concetration.dim().z);
	const auto dim = mask.dim();
	
	const auto totalElem = dim.x * dim.y * dim.z;


	const float * concData = (float *)concetration.getCurrentPtr().getCPU();
	const uchar * cdata = (uchar *)mask.getCurrentPtr().getCPU();

	int zeroElem = 0;
	for (auto i = 0; i < totalElem; i++) {
		zeroElem += (cdata[i] == 0) ? 1 : 0;
	}
	float porosity = zeroElem / float(totalElem);


	

	const int primaryDim = getDirIndex(dir);
	const int secondaryDims[2] = { (primaryDim + 1) % 3, (primaryDim + 2) % 3 };
	
	float sum = 0.0f;
	float sumHigh = 0.0f;
	int n = 0;	
	int k = (getDirSgn(dir) == -1) ? 0 : dim[primaryDim] - 1;
	int kHigh = (getDirSgn(dir) == 1) ? 0 : dim[primaryDim] - 1;

	for (auto i = 0; i < dim[secondaryDims[0]]; i++) {
		for (auto j = 0; j < dim[secondaryDims[1]]; j++) {
			ivec3 pos;
			pos[primaryDim] = k;
			pos[secondaryDims[0]] = i;
			pos[secondaryDims[1]] = j;
			
			sum += concData[linearIndex(dim, pos)];
			
			pos[primaryDim] = kHigh;
			sumHigh += concData[linearIndex(dim, pos)];

			n++;
		}
	}

	const float d0 = 0.001f;

	

	float avgJ = sum / n;
	//float h = 1.0f / dim[primaryDim];	

	float high = 1.0f;
	float low = 0.0f;
	float dc = high - low;
	float dx = 1.0f;

	float tau = d0 * porosity * dc / (avgJ * dx);

	std::cout << "Deff: " << (avgJ * dx) / dc << std::endl;
	std::cout << "porosity: " << porosity << std::endl;

	//float t = (d0 * (1.0f * porosity) / dc);
	//float t2 = h * porosity / dc / 1.0f / 2.0f;

	return tau;

	

	
}

