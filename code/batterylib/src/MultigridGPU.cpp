#include "MultigridGPU.h"

#include "cuda/Volume.cuh"
#include "cuda/MultigridGPU.cuh"

using namespace blib;

template class MultigridGPU<float>;
template class MultigridGPU<double>;

//#define MG_LINSYS_TO_FILE

#include <iostream>
#include <string>

#ifdef MG_LINSYS_TO_FILE
#include <fstream>
#endif


uint3 make_uint3(ivec3 v) {
	return make_uint3(v.x, v.y, v.z);
}


template <typename T>
MultigridGPU<T>::MultigridGPU(bool verbose)
	: _verbose(verbose),
	_type((std::is_same<float, T>::value) ? TYPE_FLOAT : TYPE_DOUBLE),
	_debugVolume(nullptr)
{


}

template <typename T>
bool ::MultigridGPU<T>::prepare(	
	const VolumeChannel & mask, 
	const VolumeChannel & concetration, 
	Dir dir, 
	T d0, T d1, 
	uint levels, 
	vec3 cellDim
)
{

	
	_cellDim = cellDim;
	_iterations = 0;
	_lv = levels;
	_dir = dir;

	
	bool allocOnCPU = false;
#ifdef MG_LINSYS_TO_FILE
	allocOnCPU = true;
#endif
	

	//Prepare lin. system for all levels
	_A.resize(_lv);
	_f.resize(_lv);
	_x.resize(_lv);
	_tmpx.resize(_lv);
	_r.resize(_lv);
	_dims.resize(_lv);	
	_D.resize(_lv);


	_streams.resize(_lv);
	for (uint i = 0; i < _lv; i++) {
		cudaStreamCreate(&_streams[i]);
	}


	//Calculate dimensions for all levels
	_dims[0] = mask.dim();	
	for (uint i = 1; i < _lv; i++) {
		ivec3 dim = _dims[i - 1] / 2;
		if (_dims[i - 1].x % 2 == 1) dim.x++;
		if (_dims[i - 1].y % 2 == 1) dim.y++;
		if (_dims[i - 1].z % 2 == 1) dim.z++;
		_dims[i] = dim;
	}	

	//Convert mask to float/double 
	{		
		auto & firstLevel = _D[0];
		firstLevel.allocOpenGL(_type, _dims[0], allocOnCPU);
		
		launchConvertMaskKernel(
			_type,
			make_uint3(_dims[0].x, _dims[0].y, _dims[0].z),
			mask.getCurrentPtr().getSurface(), 
			firstLevel.getSurface(),
			d0, d1
		);					

		/*if (_debugVolume) {
			cudaDeviceSynchronize();
			_debugVolume->emplaceChannel(
				VolumeChannel(firstLevel, _dims[0], "D in float/double")
			);
		}*/
	}

	//Restrict diffusion coeffs for smaller grids
	for (uint i = 1; i < _lv; i++) {
				
		_D[i].allocOpenGL(_type, _dims[i], allocOnCPU);
				
		launchRestrictionKernel(
			_type,
			_D[i - 1].getSurface(),
			make_uint3(_dims[i - 1].x, _dims[i - 1].y, _dims[i - 1].z),
			_D[i].getSurface(),
			make_uint3(_dims[i].x, _dims[i].y, _dims[i].z),
			T(0.5)
		);

		/*if (_debugVolume) {
			cudaDeviceSynchronize();
			_debugVolume->emplaceChannel(
				VolumeChannel(_D[i], _dims[i])
			);
		}*/
	}


	//Build matrices/vectors
	for (uint i = 0; i < _lv; i++) {
		//TODO: different streams, can be concurrent
		prepareSystemAtLevel(i);
	}
	cudaDeviceSynchronize();
	

	{
		const size_t maxN = _dims[0].x * _dims[0].y * _dims[0].z;
		const size_t reduceN = maxN / VOLUME_REDUCTION_BLOCKSIZE;
		_auxReduceBuffer.allocHost(reduceN, primitiveSizeof(_type));
		_auxReduceBuffer.allocDevice(reduceN, primitiveSizeof(_type));
	}
	


	return true;
}


template <typename T>
void blib::MultigridGPU<T>::prepareSystemAtLevel(uint level)
{
	const auto valPerRow = 7;

	auto dim = _dims[level];
	size_t N = dim.x * dim.y * dim.z;
	
	

	bool allocOnCPU = false;
#ifdef MG_LINSYS_TO_FILE
	allocOnCPU = true;
#endif

	
	_A[level].allocDevice(
		N*valPerRow,
		primitiveSizeof(_type)
	);

	if (allocOnCPU) {
		_A[level].allocHost(
			N*valPerRow,
			primitiveSizeof(_type)
		);
	}


	_f[level].allocOpenGL(_type, dim, allocOnCPU);
	_x[level].allocOpenGL(_type, dim, allocOnCPU);
	_r[level].allocOpenGL(_type, dim, allocOnCPU);
	_tmpx[level].allocOpenGL(_type, dim, allocOnCPU);

	


	LinSysParams params;
	params.type = _type;
	params.res = make_uint3(dim.x, dim.y, dim.z);
	params.cellDim = make_float3(_cellDim.x, _cellDim.y, _cellDim.z);
	params.dir = _dir;
	params.dirPrimary = getDirIndex(_dir);
	params.dirSecondary = make_uint2((params.dirPrimary + 1) % 3, (params.dirPrimary + 2) % 3 );
	params.matrixData = _A[level].gpu;
	params.surfX = _x[level].getSurface();
	params.surfF = _f[level].getSurface();
	params.surfD = _D[level].getSurface();
	
	
	launchPrepareSystemKernel(params);



	if (_verbose) {
		const auto mb = [](size_t bytes) -> float {return bytes / float(1024 * 1024); };
		std::cout << "Prepared level " << level << std::endl;
		std::cout << "Dim " << dim.x << " x " << dim.y << " x " << dim.z << std::endl;
		std::cout << "A: " << N << "x" << N << ", " << mb(_A[level].byteSize()) << "MB" << std::endl;
		std::cout << "f: " << mb(_f[level].byteSize()) << "MB" << std::endl;
		std::cout << "x: " << mb(_x[level].byteSize()) << "MB" << std::endl;
		std::cout << "r: " << mb(_r[level].byteSize()) << "MB" << std::endl;
	}


#ifdef MG_LINSYS_TO_FILE
	
	const ivec3 stride = { 1, dim.x, dim.x*dim.y };

	{		
		_A[level].retrieve();

		const int offset[7] = {
			-stride.z,
			-stride.y,
			-stride.x,
			0,
			stride.x,
			stride.y,
			stride.z
		};

		char buf[24]; itoa(level, buf, 10);
		std::ofstream f("A_GPU_" + std::string(buf) + ".dat");

		for (auto i = 0; i < N; i++) {
			T * rowPtr = ((T*)_A[level].cpu) + i*7;
			for (auto k = 0; k < 7; k++) {			

				auto j = i + offset[k];
				if (j < 0 || j >= N || rowPtr[k] == T(0)) continue;

				f << i << " " << j << " " << rowPtr[k] << "\n";
			}

			if (N < 100 || i % (N / 100))
				f.flush();
		}
	}

	{
		
		_f[level].retrieve();

		T * ptr = (T*)(_f[level].getCPU());

		char buf[24]; itoa(level, buf, 10);
		std::ofstream f("B_GPU_" + std::string(buf) + ".txt");
		for (auto i = 0; i < N; i++) {
			f << ptr[i] << "\n";
			if (N < 100 || i % (N / 100))
				f.flush();
		}

	}

	{

		_D[level].retrieve();

		T * ptr = (T*)(_D[level].getCPU());

		char buf[24]; itoa(level, buf, 10);
		std::ofstream f("D_GPU_" + std::string(buf) + ".txt");
		for (auto i = 0; i < N; i++) {
			f << ptr[i] << "\n";
			if (N < 100 || i % (N / 100))
				f.flush();
		}

	}

#endif

}

template <typename T>
T blib::MultigridGPU<T>::squareNorm(Texture3DPtr & surf, ivec3 dim)
{
	T res = T(0.0);
	launchReduceKernel(
		_type,
		REDUCE_OP_SQUARESUM,
		make_uint3(_dims[0]),
		surf.getSurface(),
		_auxReduceBuffer.gpu,
		_auxReduceBuffer.cpu,
		&res
	);
	return res;
}


template <typename T>
T MultigridGPU<T>::solve(T tolerance, size_t maxIterations, CycleType cycleType)
{

	

	const int preN = 1;
	const int postN = 1;
	const int lastLevel = _lv - 1;

	const std::vector<int> cycle = genCycle(cycleType, _lv);

	const auto tabs = [](int n) {
		return std::string(n, '\t');
	};


	//
	//r = f - Ax
	residual(
		_type,
		make_uint3(_dims[0]),
		_r[0].getSurface(),
		_f[0].getSurface(),
		_x[0].getSurface(),
		_A[0].gpu
	);

	
	T lastError = sqrt(
		squareNorm(_r[0], _dims[0]) / squareNorm(_f[0], _dims[0])
	);

	std::cout << "inital error: " << lastError << std::endl;
	


	if (_debugVolume) {
		cudaDeviceSynchronize();
		auto i = 0;
		_debugVolume->emplaceChannel(VolumeChannel(_r[i], _dims[i],"r"));
		_debugVolume->emplaceChannel(VolumeChannel(_x[i], _dims[i], "x"));
		_debugVolume->emplaceChannel(VolumeChannel(_f[i], _dims[i], "f"));
	}



	//need surface sub, add, setToZero
	//matrixdata * surface multiply, residual r - A*x
	//weighted interpolation
	//surface squared norm
	//gauss seidel zebra pass - similar to A*x

	for (auto k = 0; k < maxIterations; k++) {

		int prevI = -1;
		for (auto i : cycle) {

			//Last level
			if (i == _lv - 1) {
				//Direct solver
				if (_verbose) {
					std::cout << tabs(i) << "Exact solve at Level " << lastLevel << std::endl;
				}
				//v[lastLevel] = exactSolver.solve(f[lastLevel]);
			}
			//Restrict
			else if (i > prevI) {

				if (i > 0) {
					T zero = 0;
					clearSurface(_type, _x[i].getSurface(), make_uint3(_dims[i]), &zero);
				}

				T err = T(0);
				GaussSeidelParams gsp;
				gsp.type = _type;
				gsp.matrixData = _A[i].gpu;
				gsp.surfB = _f[i].getSurface();
				gsp.surfX = _x[i].getSurface();
				gsp.surfR = _r[i].getSurface();
				gsp.res = make_uint3(_dims[i]);
				gsp.errorOut = &err;
				gsp.tolerance = &tolerance;
				gsp.auxBufferCPU = _auxReduceBuffer.cpu;
				gsp.auxBufferGPU = _auxReduceBuffer.gpu;
				gsp.maxIter = preN;

				solveGaussSeidel(gsp);

				//std::cout << err << std::endl;

				launchWeightedRestrictionKernel(
					_type,
					_r[i].getSurface(),
					_D[i].getSurface(),
					make_uint3(_dims[i]),
					_f[i + 1].getSurface(),
					make_uint3(_dims[i + 1])
				);

			}			
			//Prolongate
			else {

				launchWeightedInterpolationKernel(
					_type,
					_x[i+1].getSurface(),
					_D[i+1].getSurface(),
					make_uint3(_dims[i]),
					_tmpx[i].getSurface(),
					make_uint3(_dims[i])
				);

				surfaceAddition(
					_type,
					_x[i].getSurface(),
					_tmpx[i].getSurface(),
					make_uint3(_dims[i])
					);


			}


		}



	}


	return 0;
}


template <typename T> 
std::vector<int> blib::MultigridGPU<T>::genCycle(CycleType ctype, uint levels)
{
	std::vector<int> cycle;

	if (ctype == V_CYCLE) {
		for (auto i = 0; i != levels; i++) {
			cycle.push_back(i);
		}
		for (auto i = levels - 2; i != -1; i--) {
			cycle.push_back(i);
		}
	}
	else if (ctype == W_CYCLE) {
		auto midLevel = (levels - 1) - 2;
		for (auto i = 0; i != levels; i++) {
			cycle.push_back(i);
		}

		for (auto i = levels - 2; i != (midLevel - 1); i--) {
			cycle.push_back(i);
		}
		for (auto i = midLevel + 1; i != levels; i++) {
			cycle.push_back(i);
		}

		for (auto i = levels - 2; i != -1; i--) {
			cycle.push_back(i);
		}

	}

	return cycle;
}