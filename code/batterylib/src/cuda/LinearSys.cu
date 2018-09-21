#include "LinearSys.cuh"

#include <stdio.h>
#include <assert.h>

__device__ __constant__ LinearSys_SysParams const_sys_params;
LinearSys_SysParams const_sys_params_cpu;


bool LinearSys_commitSysParams(const LinearSys_SysParams & sysparams) {
	cudaError_t res = cudaMemcpyToSymbol(
		const_sys_params,
		&sysparams,
		sizeof(LinearSys_SysParams),
		0,
		cudaMemcpyHostToDevice
	);

	const_sys_params_cpu = sysparams;

	return res == cudaSuccess;
}



template <typename T>
__global__ void ___generateDomain(
	const CUDA_Volume binaryMask,
	double value_zero,
	double value_one,
	CUDA_Volume output
) {
	VOLUME_VOX_GUARD(output.res);

	//Read mask
	uchar c = read<uchar>(binaryMask.surf, vox);

	//Write value
	write<T>(output.surf, vox, (c > 0) ? T(value_one) : T(value_zero));
}



void LinearSys_GenerateDomain(
	const CUDA_Volume & binaryMask,
	double value_zero,
	double value_one,
	CUDA_Volume & output
) {

	BLOCKS3D(8, output.res);

	if (binaryMask.type != TYPE_UCHAR) {
		exit(1);
	}

	if (output.type == TYPE_DOUBLE) {
		___generateDomain<double> << < numBlocks, block >> > (
			binaryMask,
			value_zero,
			value_one,
			output
			);
	}
	else if(output.type == TYPE_FLOAT) {
		___generateDomain<float> << < numBlocks, block >> > (
			binaryMask,
			value_zero,
			value_one,
			output
			);
	}
	else {
		exit(2);
	}



}




// Lin.sys at top level
template <typename T>
__device__ void getSystemTopKernel (
	const CUDA_Volume & domain,
	const uint3 & vox,
	CUDA_Stencil_7 * out,
	T * f,
	T * x
) {

	T Di = read<T>(domain.surf, vox);

	T Dneg[3] = {
		(read<T>(domain.surf, clampedVox(domain.res, vox, X_NEG)) + Di) * T(0.5),
		(read<T>(domain.surf, clampedVox(domain.res, vox, Y_NEG)) + Di) * T(0.5),
		(read<T>(domain.surf, clampedVox(domain.res, vox, Z_NEG)) + Di) * T(0.5)
	};
	T Dpos[3] = {
		(read<T>(domain.surf, clampedVox(domain.res, vox, X_POS)) + Di) * T(0.5),
		(read<T>(domain.surf, clampedVox(domain.res, vox, Y_POS)) + Di) * T(0.5),
		(read<T>(domain.surf, clampedVox(domain.res, vox, Z_POS)) + Di) * T(0.5)
	};


	T coeffs[7];
	bool useInMatrix[7];

	coeffs[DIR_NONE] = T(0);
	useInMatrix[DIR_NONE] = true;

	for (uint j = 0; j < DIR_NONE; j++) {
		const uint k = _getDirIndex(Dir(j));
		const int sgn = _getDirSgn(Dir(j));
		const T Dface = (sgn == -1) ? Dneg[k] : Dpos[k];

		T cellDist[3] = { T(const_sys_params.cellDim[0]) , T(const_sys_params.cellDim[1]), T(const_sys_params.cellDim[2]) };
		useInMatrix[j] = true;

		if ((_at<uint>(vox, k) == 0 && sgn == -1) ||
			(_at<uint>(vox, k) == _at<uint>(domain.res, k) - 1 && sgn == 1)
			) {
			cellDist[k] = const_sys_params.cellDim[k] * T(0.5);
			useInMatrix[j] = false;
		}

		coeffs[j] = (Dface * const_sys_params.faceArea[k]) / cellDist[k];

		//Subtract from diagonal
		if (useInMatrix[j] || k == const_sys_params.dirPrimary)
			coeffs[DIR_NONE] -= coeffs[j];
	}


	/*
	Calculate right hand side
	*/
	const uint primaryRes = ((uint*)&domain.res)[const_sys_params.dirPrimary];
	T rhs = T(0);
	if (_at<uint>(vox, const_sys_params.dirPrimary) == 0) {
		Dir dir = _getDir(const_sys_params.dirPrimary, -1);
		rhs -= coeffs[dir] * const_sys_params.concetrationBegin;
	}
	else if (_at<uint>(vox, const_sys_params.dirPrimary) == primaryRes - 1) {
		Dir dir = _getDir(const_sys_params.dirPrimary, 1);
		rhs -= coeffs[dir] * const_sys_params.concetrationEnd;
	}

	*f = rhs;

	/*
	Initial guess
	*/
	uint primaryVox = _at<uint>(vox, const_sys_params.dirPrimary);
	if (_getDirSgn(const_sys_params.dir) == 1)
		*x = 1.0f - (primaryVox / T(primaryRes + 1));
	else
		*x = (primaryVox / T(primaryRes + 1));



#pragma unroll
	for (uint j = 0; j < DIR_NONE; j++) {
		if (!useInMatrix[j])
			coeffs[j] = T(0);
	}

#pragma unroll
	for (uint i = 0; i < 7; i++) {
		out->v[i] = coeffs[i];
	}




}


template<typename T>
void __global__ ___systemTopKernel(
	CUDA_Volume domain,
	CUDA_Stencil_7 * A,
	CUDA_Volume f
) {
	VOLUME_VOX_GUARD(domain.res);

	size_t i = _linearIndex(domain.res, vox);

	T fval = 0.0;
	T xval = 0.0;
	getSystemTopKernel<T>(domain, vox, &A[i], &fval, &xval);
	write<T>(f.surf, vox, fval);

}

template<typename T>
void __global__ ___systemTopKernelWithGuess(
	CUDA_Volume domain,
	CUDA_Stencil_7 * A,
	CUDA_Volume f,
	CUDA_Volume x
) {
	VOLUME_VOX_GUARD(domain.res);

	size_t i = _linearIndex(domain.res, vox);

	T fval = 0.0;
	T xval = 0.0;
	getSystemTopKernel<T>(domain, vox, &A[i], &fval, &xval);
	write<T>(f.surf, vox, fval);
	write<T>(x.surf, vox, xval);

}

void LinearSys_GenerateSystem(
	const CUDA_Volume & domain,
	CUDA_Stencil_7 * A0,
	CUDA_Volume & f,
	CUDA_Volume * x
) {

	assert(domain.type == f.type);
	if (x) {
		assert(domain.type == x->type);
	}

	assert(domain.type == TYPE_DOUBLE || domain.type == TYPE_FLOAT);
	

	BLOCKS3D(2, domain.res);
	if (x == nullptr) {
		if (domain.type == TYPE_DOUBLE) 
			___systemTopKernel<double><< < numBlocks, block >> > (domain,A0,f);
		else
			___systemTopKernel<float> << < numBlocks, block >> > (domain, A0, f);
	}
	else {
		if (domain.type == TYPE_DOUBLE)
			___systemTopKernelWithGuess <double> << < numBlocks, block >> > (domain, A0, f, *x);
		else
			___systemTopKernelWithGuess <float> << < numBlocks, block >> > (domain, A0, f, *x);		
	}



}



template <typename T>
__global__ void ___invertDiagKernel(
	const CUDA_Stencil_7 * A,
	CUDA_Volume ainvert
) {
	VOLUME_VOX_GUARD(ainvert.res);
	const size_t I = _linearIndex(ainvert.res, vox);
	const T iadiag = T(1.0) / A[I].v[DIR_NONE];
	write<T>(ainvert.surf, vox, iadiag);
}


void LinearSys_InvertSystemDiagTo(
	const CUDA_Stencil_7 * A,
	CUDA_Volume & ainvert
) {

	assert(ainvert.type == TYPE_DOUBLE || ainvert.type == TYPE_FLOAT);

	BLOCKS3D(4, ainvert.res);
	
	if (ainvert.type == TYPE_DOUBLE) {
		___invertDiagKernel<double> << < numBlocks, block >> > (A, ainvert);
	}
	else {
		___invertDiagKernel<float> << < numBlocks, block >> > (A, ainvert);
	}
}




__global__ void __residualKernel(	
	const CUDA_Stencil_7 * A0,
	const CUDA_Volume x,
	const CUDA_Volume f,
	CUDA_Volume r
) {
	VOLUME_IVOX_GUARD(x.res);

	const size_t I = _linearIndex(x.res, ivox);

	const double Axval = convolve3D_SystemTop(ivox, A0[I], x);
	const double fval = read<double>(f.surf, ivox);
	const double rval = fval - Axval;

	write<double>(r.surf, ivox, rval);

}


void LinearSys_Residual(	
	const CUDA_Stencil_7 * A0,
	const CUDA_Volume & x,
	const CUDA_Volume & f,
	CUDA_Volume & r
) {
	BLOCKS3D(8, x.res);
	__residualKernel << < numBlocks, block >> > (A0, x, f, r);
}


__global__ void ___matrixVectorProductKernel(
	const CUDA_Stencil_7 * A,
	const CUDA_Volume x,
	CUDA_Volume b
) {
	VOLUME_IVOX_GUARD(x.res);
	const size_t I = _linearIndex(x.res, ivox);
	const double Axval = convolve3D_SystemTop(ivox, A[I], x);
	write<double>(b.surf, ivox, Axval);
}



template<int blockSize, int apronSize>
double __device__  convolve3D_SystemTop_Shared(
	const int3 & relVox, //in block relative voxel
	const CUDA_Stencil_7 & k,
	double * x
) {
	double sum = 0.0;

	//Index in apron
	const int index = _linearIndex(make_int3(apronSize), relVox + make_int3(1));

	const int3 stride = make_int3(1, apronSize, apronSize*apronSize);
	sum += k.v[X_POS] * x[index + stride.x];
	sum += k.v[X_NEG] * x[index - stride.x];
	sum += k.v[Y_POS] * x[index + stride.y];
	sum += k.v[Y_NEG] * x[index - stride.y];
	sum += k.v[Z_POS] * x[index + stride.z];
	sum += k.v[Z_NEG] * x[index - stride.z];
	sum += k.v[DIR_NONE] * x[index];

	return sum;
}

template <int blockSize>
__global__ void ___matrixVectorProductKernelShared(
	const CUDA_Stencil_7 * A,
	const CUDA_Volume x,
	CUDA_Volume b
) {


	const int apronSize = blockSize + 2;
	const int totalBlockSize = blockSize*blockSize*blockSize;
	const int totalApronSize = apronSize * apronSize * apronSize;
	const int perThread = (totalApronSize + totalBlockSize - 1) / totalBlockSize;

	__shared__ double _s[totalApronSize];

	VOLUME_BLOCK_IVOX; //defines blockIvox

					   //Position of apron
	const int3 apronIvox = blockIvox - make_int3(1);

	//Load apron to shared memory
	int tid = _linearIndex(blockDim, threadIdx);

	/*size_t blockIndex = _linearIndex(gridDim, blockIdx);
	if (blockIndex == 1) {
	printf("tid: %d, <%d, %d)\n", tid, min(tid*perThread, totalApronSize), min((tid+1)*perThread, totalApronSize));
	}*/

	//	int3 relativeLoadVoxel = apronIvox + 
	for (int i = 0; i < perThread; i++) {
		const int targetIndex = tid + i*totalBlockSize;
		//const int targetIndex = tid*perThread + i;
		if (targetIndex >= totalApronSize)
			break;

		const int3 targetPos = apronIvox + posFromLinear(make_int3(apronSize), targetIndex);
		if (_isValidPos(x.res, targetPos))
			_s[targetIndex] = read<double>(x.surf, targetPos);
		else
			_s[targetIndex] = 0.0;
	}

	__syncthreads();



	//VOLUME_IVOX_GUARD(x.res);
	//write<double>(b.surf, ivox, 1.0);

	VOLUME_IVOX_GUARD(x.res);
	const size_t I = _linearIndex(x.res, ivox);
	CUDA_Stencil_7 a = A[I];
	const double Axval = convolve3D_SystemTop_Shared<blockSize, apronSize>(make_int3(threadIdx.x, threadIdx.y, threadIdx.z), a, _s);

	//const double Axval = convolve3D_SystemTop(ivox, A[I], x);
	write<double>(b.surf, ivox, Axval);

}



void LinearSys_MatrixVectorProduct(
	const CUDA_Stencil_7 * A,
	const CUDA_Volume & x,
	CUDA_Volume & b
) {

	const bool optimized = true;
	if (!optimized) {
		BLOCKS3D(8, x.res);
		___matrixVectorProductKernel << < numBlocks, block >> > (A, x, b);
	}
	else {
		const int blockSize = 8;
		BLOCKS3D(blockSize, x.res);
		___matrixVectorProductKernelShared<blockSize> << < numBlocks, block >> > (A, x, b);
	}
}