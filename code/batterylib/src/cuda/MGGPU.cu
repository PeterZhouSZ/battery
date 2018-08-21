#include "MGGPU.cuh"


#define MAX_CONST_KERNEL_DIM 7
__device__ __constant__ double const_kernel[MAX_CONST_KERNEL_DIM * MAX_CONST_KERNEL_DIM * MAX_CONST_KERNEL_DIM];
__device__ __constant__ int const_kernel_dim;

__device__ __constant__ MGGPU_SysParams const_sys_params;

bool commitSysParams(const MGGPU_SysParams & sysparams) {
	cudaError_t res = cudaMemcpyToSymbol(
		const_sys_params, 
		&sysparams, 
		sizeof(MGGPU_SysParams),
		0,
		cudaMemcpyHostToDevice
	);
	return res == cudaSuccess;
}

__global__ void ___generateDomain(
	const MGGPU_Volume binaryMask,
	double value_zero,
	double value_one,
	MGGPU_Volume output
) {
	VOLUME_VOX_GUARD(output.res);	

	//Read mask
	uchar c = read<uchar>(binaryMask.surf, vox);		

	//Write value
	write<double>(output.surf, vox, (c > 0) ? value_one : value_zero);
}



void MGGPU_GenerateDomain(
	const MGGPU_Volume & binaryMask,
	double value_zero,
	double value_one,
	MGGPU_Volume & output
) {

	BLOCKS3D(2, output.res);	
	___generateDomain<< < numBlocks, block >> > (
		binaryMask,
		value_zero, 
		value_one, 
		output
	);
}


//Kernel for 3D convolution by 2^3 kernel with stride 2
__global__ void ___convolve3D_2_2(
	const MGGPU_Volume in,	
	const MGGPU_Volume out
){
	//todo shared mem
	VOLUME_VOX_GUARD(out.res);

	double sum = 0.0;

	MGGPU_Kernel3D<2> & k = *((MGGPU_Kernel3D<2>*)const_kernel);
	
	//TODO: boundary handling (upper only)
	sum += read<double>(in.surf, 2 * vox + make_uint3(0, 0, 0)) * k.v[0][0][0];	
	sum += read<double>(in.surf, 2 * vox + make_uint3(0, 0, 1)) * k.v[0][0][1];
	sum += read<double>(in.surf, 2 * vox + make_uint3(0, 1, 0)) * k.v[0][1][0];
	sum += read<double>(in.surf, 2 * vox + make_uint3(0, 1, 1)) * k.v[0][1][1];
	sum += read<double>(in.surf, 2 * vox + make_uint3(1, 0, 0)) * k.v[1][0][0];
	sum += read<double>(in.surf, 2 * vox + make_uint3(1, 0, 1)) * k.v[1][0][1];
	sum += read<double>(in.surf, 2 * vox + make_uint3(1, 1, 0)) * k.v[1][1][0];
	sum += read<double>(in.surf, 2 * vox + make_uint3(1, 1, 1)) * k.v[1][1][1];

	write<double>(out.surf, vox, sum);

}

void MGGPU_Convolve(
	const MGGPU_Volume & in,
	MGGPU_KernelPtr kernel, int kn,
	const MGGPU_Volume & out
) {

	cudaError_t res0 = cudaMemcpyToSymbol(const_kernel, kernel, sizeof(double) * kn * kn * kn, 0, cudaMemcpyHostToDevice);
	cudaError_t res1 = cudaMemcpyToSymbol(const_kernel_dim, &kn, sizeof(int), 0, cudaMemcpyHostToDevice);
	

	
	BLOCKS3D(2, out.res);
	if (kn == 2) {
		
		___convolve3D_2_2<<< numBlocks, block >>> (
			in,			
			out
			);
	}

}



template <typename T>
__device__ void MGGPU_GetSystemTopKernel(
	const MGGPU_Volume & domain,
	const uint3 & vox,
	MGGPU_SystemTopKernel * out,
	T * f = nullptr
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

		T cellDist[3] = { const_sys_params.cellDim[0],const_sys_params.cellDim[1],const_sys_params.cellDim[2] };
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


	if (f != nullptr) {
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
	}

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

void __global__ ___systemTopKernel(
	MGGPU_Volume domain,
	MGGPU_SystemTopKernel * A0,
	MGGPU_Volume f
){
	VOLUME_VOX_GUARD(domain.res);

	size_t i = _linearIndex(domain.res, vox);

	double fval = 0.0;
	MGGPU_GetSystemTopKernel<double>(domain, vox, &A0[i], &fval);	
	write<double>(f.surf, vox, fval);

}

void MGGPU_GenerateSystemTopKernel(
	const MGGPU_Volume & domain,
	MGGPU_SystemTopKernel * A0,
	MGGPU_Volume & f
) {

	BLOCKS3D(2, domain.res);
	___systemTopKernel << < numBlocks, block >> > (
		domain,
		A0,
		f
		);
	


}