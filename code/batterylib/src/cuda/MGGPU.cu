#include "MGGPU.cuh"
#include <stdio.h>

#define MAX_CONST_KERNEL_DIM 7

struct KernelCombineParams {
	uint3 resArow;
	uint3 resAcol;
	uint3 resBrow;
	uint3 resBcol;
	uint3 resCrow;
	uint3 resCcol;
	MGGPU_KernelPtr A,B,C;	
	int Adim, Bdim, Cdim;	
	int Aratio, Bratio, Cratio;
	int AdimHalf, BdimHalf, CdimHalf;

	MGGPU_Volume domain;
};


__device__ __constant__ double const_kernel[MAX_CONST_KERNEL_DIM * MAX_CONST_KERNEL_DIM * MAX_CONST_KERNEL_DIM];
__device__ __constant__ int const_kernel_dim;

__device__ __constant__ MGGPU_SysParams const_sys_params;
MGGPU_SysParams const_sys_params_cpu;

__device__ __constant__ KernelCombineParams const_kernel_combine_params;




bool commitSysParams(const MGGPU_SysParams & sysparams) {
	cudaError_t res = cudaMemcpyToSymbol(
		const_sys_params, 
		&sysparams, 
		sizeof(MGGPU_SysParams),
		0,
		cudaMemcpyHostToDevice
	);

	const_sys_params_cpu = sysparams;

	return res == cudaSuccess;
}

bool commitKernelCombineParams(const KernelCombineParams & p) {
	cudaError_t res = cudaMemcpyToSymbol(
		const_kernel_combine_params,
		&p,
		sizeof(KernelCombineParams),
		0,
		cudaMemcpyHostToDevice
	);
	if (res != cudaSuccess) {
		printf("Failed to commit kernel combine params\n");
		exit(1);
	}
	return res == cudaSuccess;
}

////////////////////////////////////////////// Generate domain

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

////////////////////////////////////////////// Convolve


//Kernel for 3D convolution by 2^3 kernel with stride 2 and border
__global__ void ___convolve3D_2_2(
	const MGGPU_Volume in,	
	const MGGPU_Volume out
){
	//todo shared mem
	VOLUME_VOX_GUARD(out.res);

	double sum = 0.0;

	MGGPU_Kernel3D<2> & k = *((MGGPU_Kernel3D<2>*)const_kernel);
		
	int3 s = make_int3(1, 1, 1); //stride
	const uint3 voxSrc = vox * 2;
	if (voxSrc.x == in.res.x - 1)
		s.x = 0;
	if (voxSrc.y == in.res.y - 1)
		s.y = 0;
	if (voxSrc.z == in.res.z - 1)
		s.z = 0;

	sum += read<double>(in.surf, voxSrc + make_uint3(0, 0, 0)) * k.v[0][0][0];
	sum += read<double>(in.surf, voxSrc + make_uint3(0, 0, s.z)) * k.v[0][0][1];
	sum += read<double>(in.surf, voxSrc + make_uint3(0, s.y, 0)) * k.v[0][1][0];
	sum += read<double>(in.surf, voxSrc + make_uint3(0, s.y, s.z)) * k.v[0][1][1];
	sum += read<double>(in.surf, voxSrc + make_uint3(s.x, 0, 0)) * k.v[1][0][0];
	sum += read<double>(in.surf, voxSrc + make_uint3(s.x, 0, s.z)) * k.v[1][0][1];
	sum += read<double>(in.surf, voxSrc + make_uint3(s.x, s.y, 0)) * k.v[1][1][0];
	sum += read<double>(in.surf, voxSrc + make_uint3(s.x, s.y, s.z)) * k.v[1][1][1];

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
	else {
		printf("MGGPU_Convolve does not support kn!=2 at the moment\n");
		exit(1);
	}

}


/*
	Kernel generation
*/


////////////////////////////////////////////// Top level system kernel

// Lin.sys at top level
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


////////////////////////////////////////////// Interpolation kernel




__device__ __host__  MGGPU_InterpKernel MGGPU_GetInterpolationKernel(
	const MGGPU_Volume & domainSrc,
	const int3 & vox, //vox in destination
	const uint3 & destRes, //should be exactly double (if power of 2) of domain.res
	int dirIndex
) {

	MGGPU_InterpKernel kernel;

	/*
	Two spaces:
	source : n/2 (domain, domain.res)
	dest: n (vox, destRes)
	*/



	memset(&kernel, 0, sizeof(MGGPU_InterpKernel));

	//If outside, return zero kernel
	if (!_isValidPos(destRes, vox)) {
		return kernel;
	}



	const int3 r = make_int3(vox.x % 2, vox.y % 2, vox.z % 2) * 2 - 1;
	const int3 voxSrc = make_int3(vox.x / 2, vox.y / 2, vox.z / 2);

#ifdef DEBUG
	if (!_isValidPos(domainSrc.res, voxSrc)) {
		printf("%d %d %d\n", voxSrc.x, voxSrc.y, voxSrc.z);
	}
#endif


	//Different offset for each subcell
	const int3 offsets[8] = {
		make_int3(0,0,0),
		make_int3(r.x,0,0),
		make_int3(0,r.y,0),
		make_int3(r.x,r.y,0),
		make_int3(0,0,r.z),
		make_int3(r.x,0,r.z),
		make_int3(0,r.y,r.z),
		make_int3(r.x,r.y,r.z)
	};

	//Initial weights
	double P[8] = {
		27, 9, 9, 3, 9, 3, 3, 1
	};

	if ((dirIndex != 0 && (vox.x == destRes.x - 1 || vox.x == 0))) {
		P[0] += P[1]; P[1] = 0;
		P[2] += P[3]; P[3] = 0;
		P[4] += P[5]; P[5] = 0;
		P[6] += P[7]; P[7] = 0;
	}

	if ((dirIndex != 1 && (vox.y == destRes.y - 1 || vox.y == 0))) {
		P[0] += P[2]; P[2] = 0;
		P[1] += P[3]; P[3] = 0;
		P[4] += P[6]; P[6] = 0;
		P[5] += P[7]; P[7] = 0;
	}

	if ((dirIndex != 2 && (vox.z == destRes.z - 1 || vox.z == 0))) {
		P[0] += P[4]; P[4] = 0;
		P[1] += P[5]; P[5] = 0;
		P[2] += P[6]; P[6] = 0;
		P[3] += P[7]; P[7] = 0;
	}




	double w[8] = {
		0,0,0,0,0,0,0,0
	};
	double W = 0.0;
	for (int i = 0; i < 8; i++) {
		if (P[i] == 0) continue;
		w[i] = P[i];


		int3 voxSrcNew = voxSrc + offsets[i];
		if (_isValidPos(domainSrc.res, voxSrcNew)) {
#ifdef __CUDA_ARCH__
			w[i] *= read<double>(domainSrc.surf, make_uint3(voxSrcNew)); //redundant conversion to uint, TODO better
#else
			w[i] *= ((double*)domainSrc.cpu)[_linearIndex(domainSrc.res, voxSrcNew)];
#endif
		}
		//Source voxel is outside of domain
		//P[i] > 0 then implies it's on dirichlet boundary
		//Therefore a nearest value has to be used
		else {

			//Change offset to nearest valid voxel from source
			int3 offset = offsets[i];

			_at<int, int3>(offset, dirIndex) = 0;
			if (!_isValidPos(domainSrc.res, voxSrc + offset)) {
				_at<int, int3>(offset, (dirIndex + 1) % 3) = 0;
			}

			if (!_isValidPos(domainSrc.res, voxSrc + offset)) {
				_at<int, int3>(offset, (dirIndex + 2) % 3) = 0;
			}

			//Update src vox with new offset
			voxSrcNew = voxSrc + offset;
#ifdef DEBUG
			if (!_isValidPos(domainSrc.res, voxSrc + offset)) {
				int3 p = voxSrc + offset;
				printf("%d %d %d\n", p.x, p.y, p.z);
			}
#endif

			//Read weight from source domain
#ifdef __CUDA_ARCH__
			w[i] *= read<double>(domainSrc.surf, make_uint3(voxSrcNew));
#else
			w[i] *= ((double*)domainSrc.cpu)[_linearIndex(domainSrc.res, voxSrcNew)];
#endif
		}

		W += w[i];
	}




	//Normalize weights
	for (auto i = 0; i < 8; i++) {
		w[i] /= W;
	}



	//Create 3^3 kernel
	memset(kernel.v, 0, INTERP_SIZE*INTERP_SIZE*INTERP_SIZE * sizeof(double));
	for (auto i = 0; i < 8; i++) {
		int3 kpos = make_int3(1, 1, 1) + offsets[i];
		kernel.v[kpos.x][kpos.y][kpos.z] = w[i];
	}


	return kernel;
}


__global__ void ___genI(
	const uint3 destRes,
	const MGGPU_Volume domainHalf,	
	MGGPU_InterpKernel * I
) {

	VOLUME_VOX_GUARD(destRes);	
	size_t i = _linearIndex(destRes, vox);

	const int3 ivox = make_int3(vox);
	I[i] = MGGPU_GetInterpolationKernel(domainHalf, ivox, destRes, const_sys_params.dirPrimary);


}


void MGGPU_GenerateSystemInterpKernels(
	const uint3 & destRes, 
	const MGGPU_Volume & domainHalf,
	MGGPU_InterpKernel * I
) {

	BLOCKS3D(destRes.x <= 4 ? 1 : 2, destRes);
	___genI << < numBlocks, block >> > (
		destRes,
		domainHalf,
		I
		);
}



__global__ void ___genITranpose(
	const uint3 Nres,
	const uint3 Nhalfres,
	const MGGPU_Volume domainHalf,
	MGGPU_Kernel3D<4> * output
){

		 
	VOLUME_VOX_GUARD(Nres);	
	int3 ivox = make_int3(vox);
	int3 ivoxhalf = make_int3(vox.x / 2, vox.y / 2, vox.z / 2);

	//Get kernel interpolating to ivox from Nhalfres
	MGGPU_InterpKernel kernel = MGGPU_GetInterpolationKernel(domainHalf, ivox, Nres, const_sys_params.dirPrimary);

	//Scatter kernel
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				double val = kernel.v[i][j][k];

				int3 targetHalf = ivoxhalf + make_int3(i - 1, j - 1, k - 1);
				size_t debugLinPosHalf = _linearIndex(Nhalfres, targetHalf);


				if (_isValidPos(Nhalfres, targetHalf)){
					//output index
					size_t index = _linearIndex(Nhalfres, targetHalf);

					//inoutput index
					int3 inOutputIndex = ivox - (targetHalf * 2 - 1);
					
					if (!_isValidPos(make_uint3(4), inOutputIndex)) {
						continue;
					}

					if (index == 0) {
						//printf("(%d, (%d, %d, %d) / (%d %d %d))\t %f\n", index, inOutputIndex.x, inOutputIndex.y, inOutputIndex.z, ivox.x, ivox.y, ivox.z, val);
					}
					output[index].v[inOutputIndex.x][inOutputIndex.y][inOutputIndex.z] = val;
				}
				
			}
		}
	}



}


////////////////////////////////////////////// Interpolation kernel




////////////////////////////////////////////// Combination (not used atm)

enum CombineType {
	A_IS_GENERIC,
	A_IS_TOPLEVEL,
	A_IS_RESTRICTION
};

template<CombineType combineType, size_t AKernelAllocSize = 5*5*5>
__device__ __host__ void combineKernelsAt(
	int3 ivox, 
	const KernelCombineParams & p
) {


	const int Asize = p.Adim*p.Adim*p.Adim;
	const int Bsize= p.Bdim*p.Bdim*p.Bdim;
	const int Csize= p.Cdim*p.Cdim*p.Cdim;
	
	//

	//i in matrix multiply
	size_t Crow = _linearIndex(p.resCrow, ivox); // in resCrow == resArow space
	const size_t Arow = Crow; // in resCrow == resArow space

	
	
	double kernelAStore[AKernelAllocSize];
	if (combineType == A_IS_GENERIC) {
		memcpy(kernelAStore, p.A + Asize * Arow, Asize * sizeof(double));
	}
	else if (combineType == A_IS_TOPLEVEL) {
		memcpy(kernelAStore, ((char*)p.A) + sizeof(MGGPU_SystemTopKernel) * Arow, sizeof(MGGPU_SystemTopKernel));
	}
	else if (combineType == A_IS_RESTRICTION) {		
		MGGPU_GetRestrictionKernel(make_uint3(ivox), p.resArow, const_sys_params.dirPrimary, kernelAStore);
	}

	const MGGPU_KernelPtr kernelA = kernelAStore;
	const MGGPU_KernelPtr kernelC = p.C + Csize * Crow;

	//Columns of C				
	for (int ck = -p.CdimHalf; ck < p.Cdim - p.CdimHalf; ck++) {
		for (int cj = -p.CdimHalf; cj < p.Cdim - p.CdimHalf; cj++) {
			for (int ci = -p.CdimHalf; ci < p.Cdim - p.CdimHalf; ci++) {

				//j in matrix multipy (only nonzero result cols)
				/*Get voxel pos in Crow space
				-> project ivox to Crow space, then apply kernel delta
				*/


				int3 ivoxCcol = make_int3(ivox.x / p.Cratio, ivox.y / p.Cratio, ivox.z / p.Cratio) + make_int3(ci, cj, ck);
				int3 ivoxBcol = ivoxCcol;

				if (!_isValidPos(p.resCcol, ivoxCcol)) {
					continue;
				}

				size_t Bcol = _linearIndex(p.resBcol, ivoxBcol);

				//multiply / dot product
				//Arow=Crow * Bcol=Ccol

				double sum = 0.0;

				if (combineType != A_IS_TOPLEVEL) {					
					for (int ak = -p.AdimHalf; ak < p.Adim - p.AdimHalf; ak++) {
						for (int aj = -p.AdimHalf; aj < p.Adim - p.AdimHalf; aj++) {
							for (int ai = -p.AdimHalf; ai < p.Adim - p.AdimHalf; ai++) {

								int3 ivoxDot = ivox * p.Aratio + make_int3(ai, aj, ak);
								if (!_isValidPos(p.resAcol, ivoxDot)) {
									continue;
								}

								size_t iDot = _linearIndex(p.resBrow, ivoxDot);

								double valA = 0.0;
								valA = kernelA[
									_linearIndexXFirst(make_uint3(p.Adim), make_int3(ai + p.AdimHalf, aj + p.AdimHalf, ak + p.AdimHalf))
								];


								MGGPU_KernelPtr kernelB = p.B + Bsize * iDot;
								int3 BkernelOffset = ivoxBcol - make_int3(ivoxDot.x / p.Bratio, ivoxDot.y / p.Bratio, ivoxDot.z / p.Bratio);
								int bi = BkernelOffset.x + p.BdimHalf;
								int bj = BkernelOffset.y + p.BdimHalf;
								int bk = BkernelOffset.z + p.BdimHalf;

								if (!_isValidPos(make_uint3(p.Bdim), make_int3(bi, bj, bk))) {
									continue;
								}

								double valB = kernelB[
									_linearIndexXFirst(make_uint3(p.Bdim), make_int3(bi, bj, bk))
								];
								sum += valA*valB;

							}
						}
					}
				}
				else {
					for (int dir = 0; dir <= DIR_NONE; dir++) {

						int3 ivoxDot = ivox * p.Aratio + dirVec(Dir(dir)); //make_int3(ai, aj, ak);
						if (!_isValidPos(p.resAcol, ivoxDot)) {
							continue;
						}

						

						

						double valA = kernelA[dir];						
#ifdef __CUDA_ARCH__
						MGGPU_InterpKernel kernelInterp = MGGPU_GetInterpolationKernel(p.domain, ivoxDot, p.resBrow, const_sys_params.dirPrimary);
#else
						//TODO?
						MGGPU_InterpKernel kernelInterp;
#endif
						MGGPU_KernelPtr kernelB = MGGPU_KernelPtr(&kernelInterp);

						

						int3 BkernelOffset = ivoxBcol - make_int3(ivoxDot.x / p.Bratio, ivoxDot.y / p.Bratio, ivoxDot.z / p.Bratio);
						int bi = BkernelOffset.x + p.BdimHalf;
						int bj = BkernelOffset.y + p.BdimHalf;
						int bk = BkernelOffset.z + p.BdimHalf;

						if (!_isValidPos(make_uint3(p.Bdim), make_int3(bi, bj, bk))) {
							continue;
						}


						double valB = kernelB[
							_linearIndexXFirst(make_uint3(p.Bdim), make_int3(bi, bj, bk))
						];

						/*{
							size_t iDot = _linearIndex(p.resBrow, ivoxDot);
							MGGPU_KernelPtr kernelB0 = p.B + Bsize * iDot;
							double valB0 = kernelB0[
								_linearIndexXFirst(make_uint3(p.Bdim), make_int3(bi, bj, bk))
							];

							double diff = valB0 - valB;
							if (diff > 0.0) {
								printf("%f\n", diff);
							}
						}*/

						

						sum += valA*valB;

					}
				}

				if (sum != 0.0) {
					kernelC[
						_linearIndexXFirst(make_uint3(p.Cdim), make_int3(ci + p.CdimHalf, cj + p.CdimHalf, ck + p.CdimHalf))
					] = sum;
				}

			}
		}
	}

	


}

template<CombineType combineType, size_t AKernelAllocSize = 5 * 5 * 5>
__global__ void __combineKernels()
{
	//Get voxel position
	VOLUME_VOX_GUARD(const_kernel_combine_params.resArow);
	int3 ivox = make_int3(vox);
	KernelCombineParams & p = const_kernel_combine_params;
	//Combine kernels at ivox position
	combineKernelsAt<combineType, AKernelAllocSize>(ivox, p);		

}


bool MGGPU_CombineKernelsTopLevel(
	const uint3 resA,
	const uint3 resBrow,
	const uint3 resBcol,
	const MGGPU_KernelPtr A,
	const MGGPU_KernelPtr B,
	const int Bdim, 
	MGGPU_KernelPtr C,
	const int Cdim,
	MGGPU_Volume interpDomain,
	bool onDevice
){

	if (resA.x != resBrow.x ||
		resA.y != resBrow.y ||
		resA.z != resBrow.z) {
		return false;
	}

	KernelCombineParams p;
	p.resArow = resA;
	p.resAcol = resA;
	p.resBrow = resBrow;
	p.resBcol = resBcol;

	p.resCrow = resA;
	p.resCcol = resBcol;

	p.A = A;
	p.B = B;
	p.C = C;

	p.Bratio = p.resBrow.x / p.resBcol.x;
	p.Aratio = p.resAcol.x / p.resArow.x;
	p.Cratio = p.resCrow.x / p.resCcol.x;

	p.Adim = 3;
	p.Bdim = Bdim;
	p.Cdim = Cdim;//MGGPU_outputKernelSize(p.Adim, p.Bdim, p.Bratio);

	p.CdimHalf = (p.Cdim % 2 == 0) ? p.Cdim / 2 - 1 : p.Cdim / 2;
	p.BdimHalf = (p.Bdim % 2 == 0) ? p.Bdim / 2 - 1 : p.Bdim / 2;
	p.AdimHalf = (p.Adim % 2 == 0) ? p.Adim / 2 - 1 : p.Adim / 2;
	
	p.domain = interpDomain;

	const CombineType combineType = A_IS_TOPLEVEL;
	const size_t allocA = sizeof(MGGPU_SystemTopKernel) / sizeof(double);

	if (onDevice) {
		commitKernelCombineParams(p);
		BLOCKS3D(2, p.resCrow);
		__combineKernels<combineType, allocA> << < numBlocks, block >> > ();
	}
	else {
		int3 ivox;
		for (ivox.z = 0; ivox.z < p.resCrow.z; ivox.z++) {
			for (ivox.y = 0; ivox.y < p.resCrow.y; ivox.y++) {
				for (ivox.x = 0; ivox.x < p.resCrow.x; ivox.x++) {
					combineKernelsAt<combineType, allocA>(ivox, p);
				}
			}
		}
	}

	return true;


}


bool MGGPU_CombineKernelsRestrict(
	const uint3 resArow,
	const uint3 resAcol,
	const uint3 resBrow,
	const uint3 resBcol,	
	const MGGPU_KernelPtr B,
	const int Bdim,
	MGGPU_KernelPtr C,
	bool onDevice
) {

	

	if (resAcol.x != resBrow.x ||
		resAcol.y != resBrow.y ||
		resAcol.z != resBrow.z) {
		return false;
	}

	KernelCombineParams p;
	p.resArow = resArow;
	p.resAcol = resAcol;
	p.resBrow = resBrow;
	p.resBcol = resBcol;

	p.resCrow = resArow;
	p.resCcol = resBcol;

	p.A = nullptr;
	p.B = B;
	p.C = C;

	p.Bratio = p.resBrow.x / p.resBcol.x;
	p.Aratio = p.resAcol.x / p.resArow.x;
	p.Cratio = p.resCrow.x / p.resCcol.x;

	p.Adim = RESTR_SIZE;
	p.Bdim = Bdim;	
	p.Cdim = MGGPU_outputKernelSize(p.Adim, p.Bdim, p.Bratio);

	p.CdimHalf = (p.Cdim % 2 == 0) ? p.Cdim / 2 - 1 : p.Cdim / 2;
	p.BdimHalf = (p.Bdim % 2 == 0) ? p.Bdim / 2 - 1 : p.Bdim / 2;
	p.AdimHalf = (p.Adim % 2 == 0) ? p.Adim / 2 - 1 : p.Adim / 2;

	const CombineType combineType = A_IS_RESTRICTION;
	const size_t allocA = RESTR_SIZE*RESTR_SIZE*RESTR_SIZE;


	if (onDevice) {
		commitKernelCombineParams(p);
		BLOCKS3D(4, p.resCrow);
		__combineKernels<combineType, allocA> << < numBlocks, block >> > ();
	}
	else {
		int3 ivox;
		for (ivox.z = 0; ivox.z < p.resCrow.z; ivox.z++) {
			for (ivox.y = 0; ivox.y < p.resCrow.y; ivox.y++) {
				for (ivox.x = 0; ivox.x < p.resCrow.x; ivox.x++) {
					combineKernelsAt<combineType, allocA>(ivox, p);
				}
			}
		}
	}

	return true;


}


bool MGGPU_CombineKernelsGeneric(
	const uint3 resArow,
	const uint3 resAcol,	
	const uint3 resBrow,
	const uint3 resBcol,	
	const MGGPU_KernelPtr A,
	const int Adim,
	const MGGPU_KernelPtr B,
	const int Bdim, // in resBcol
	MGGPU_KernelPtr C,
	const int Cdim,
	bool onDevice
) {
	
	if (resAcol.x != resBrow.x ||
		resAcol.y != resBrow.y ||
		resAcol.z != resBrow.z) {
		return false;
	}

	KernelCombineParams p;
	p.resArow = resArow;
	p.resAcol = resAcol;
	p.resBrow = resBrow;
	p.resBcol = resBcol;

	p.resCrow = resArow;
	p.resCcol = resBcol;

	p.A = A;
	p.B = B;
	p.C = C;
		
	p.Bratio = p.resBrow.x / p.resBcol.x;
	p.Aratio = p.resAcol.x / p.resArow.x;
	p.Cratio = p.resCrow.x / p.resCcol.x;

	p.Adim = Adim;
	p.Bdim = Bdim;
	p.Cdim = MGGPU_outputKernelSize(p.Adim, p.Bdim, p.Bratio);

	p.CdimHalf = (p.Cdim % 2 == 0) ? p.Cdim / 2 - 1 : p.Cdim / 2;
	p.BdimHalf = (p.Bdim % 2 == 0) ? p.Bdim / 2 - 1 : p.Bdim / 2;
	p.AdimHalf = (p.Adim % 2 == 0) ? p.Adim / 2 - 1 : p.Adim / 2;

	const CombineType combineType = A_IS_GENERIC;
	const size_t allocA = 5*5*5;


	if (onDevice) {
		commitKernelCombineParams(p);
		BLOCKS3D(4, p.resCrow);
		__combineKernels<combineType, allocA> << < numBlocks, block >> > ();
	}
	else {
		int3 ivox;
		for (ivox.z = 0; ivox.z < p.resCrow.z; ivox.z++) {
			for (ivox.y = 0; ivox.y < p.resCrow.y; ivox.y++) {
				for (ivox.x = 0; ivox.x < p.resCrow.x; ivox.x++) {
					combineKernelsAt<combineType, allocA>(ivox, p);
				}
			}
		}	
	}
	

	return true;
}





////////////////////////////////////////////// Galerking coarseing operator - Generating Ai

#define RA0_SIZE (RESTR_SIZE + 2)
#define RA0_HALF (2)

#define RESTR_HALF (1)

#define A1_SIZE (5)
#define A1_HALF (2)
#define A1_THREADS_PER_DIM (4)

//#define UNROLL_GENA1 #pragma unroll
#define UNROLL_GENA1 {}


__device__ __host__ void buildA1At(
	int3 & ivox,
	const uint3 & resA0,
	const uint3 & resA1,
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_InterpKernel * I,
	MGGPU_Kernel3D<5> * A1,
	int dirPrimary	
) {

	const int3 NspaceIvox = ivox * 2;
	const size_t debugI = _linearIndex(resA1, ivox);

	MGGPU_RestrictKernel R;
	MGGPU_GetRestrictionKernel(make_uint3(ivox), resA1, dirPrimary, (double*)&R);


	//Temporary result of R*A at row(ivox)
	MGGPU_Kernel3D<RA0_SIZE> RA;
	
	UNROLL_GENA1	
	for (auto rax = 0; rax < RA0_SIZE; rax++) {	
		UNROLL_GENA1
		for (auto ray = 0; ray < RA0_SIZE; ray++) {
			UNROLL_GENA1
				for (auto raz = 0; raz < RA0_SIZE; raz++) {
			

				int3 voxRA = NspaceIvox + make_int3(rax - RA0_HALF, ray - RA0_HALF, raz - RA0_HALF);
				if (!_isValidPos(resA0, voxRA)) {
					RA.v[rax][ray][raz] = 0.0;
					continue;
				}


				//ivox + rx,y,z is in N space
				double sum = 0.0;		
				for (auto rx = 0; rx < RESTR_SIZE; rx++) {
					for (auto ry = 0; ry < RESTR_SIZE; ry++) {											
						for (auto rz = 0; rz < RESTR_SIZE; rz++) {
							double rval = R.v[rx][ry][rz];							
							int3 rvox = NspaceIvox + make_int3(rx - RESTR_HALF, ry - RESTR_HALF, rz - RESTR_HALF);
							if (!_isValidPos(resA0, rvox)) {
								continue;
							}
						
							size_t NspaceIndex = _linearIndex(resA0, rvox);							

							//Top level
							{
								const MGGPU_SystemTopKernel & A = A0[NspaceIndex];
								int3 Aoffset = voxRA - rvox;								
								double aval = MGGPU_GetTopLevelValue(A, Aoffset);							
								sum += aval*rval;
							}
						}
					}
				}

				
				RA.v[rax][ray][raz] = sum;
			}
		}
	}


	
	
	size_t a1index = _linearIndex(resA1, ivox);
	MGGPU_Kernel3D<A1_SIZE> & a1 = A1[a1index];

	
	
	
	for (auto xa1 = 0; xa1 < A1_SIZE; xa1++) {
		for (auto ya1 = 0; ya1 < A1_SIZE; ya1++) {		
			for (auto za1 = 0; za1 < A1_SIZE; za1++) {
			

				int3 voxA1 = ivox + make_int3(xa1 - A1_HALF, ya1 - A1_HALF, za1 - A1_HALF);				
				if (!_isValidPos(resA0, voxA1)) {
					a1.v[xa1][ya1][za1] = 0.0;
					continue;
				}				

				
				double sum = 0.0;				
				for (auto rax = 0; rax < RA0_SIZE; rax++) {				
					for (auto ray = 0; ray < RA0_SIZE; ray++) {
						for (auto raz = 0; raz < RA0_SIZE; raz++) {						

							int3 ravox = NspaceIvox + make_int3(rax - RA0_HALF, ray - RA0_HALF, raz - RA0_HALF);
							if (!_isValidPos(resA0, ravox)) {
								continue;
							}

							double rval = RA.v[rax][ray][raz];
							size_t NspaceIndex = _linearIndex(resA0, ravox);
							
							{	

								int3 offset = voxA1 - make_int3(ravox.x / 2, ravox.y / 2, ravox.z / 2) + make_int3(INTERP_SIZE / 2);
								if (!_isValidPos(make_uint3(INTERP_SIZE), offset)) {
									continue;
								}

								double ival = I[NspaceIndex].v[offset.x][offset.y][offset.z];
								
								sum += ival*rval;
							}
						}
					}
				}

				a1.v[xa1][ya1][za1] = sum;
				
			}
		}
	}
	

	return;
}

#define RAi_SIZE 8
#define RAi_HALF 3
#define Ai_SIZE 5
#define Ai_HALF 2

#define Ai_THREADS_PER_DIM (4)


__device__ __host__ void buildAiAt(
	int3 & ivox,
	const uint3 & resAprev,
	const uint3 & resAnext,
	const MGGPU_Kernel3D<Ai_SIZE> * Aprev,
	const MGGPU_InterpKernel * I,
	MGGPU_Kernel3D<Ai_SIZE> * Anext,
	int dirPrimary
) {


	const int3 NspaceIvox = ivox * 2;	
	const size_t debugI = _linearIndex(resAnext, ivox);

	MGGPU_RestrictKernel R;
	MGGPU_GetRestrictionKernel(make_uint3(ivox), resAnext, dirPrimary, (double*)&R);


	//Temporary result of R*A at row(ivox)
	MGGPU_Kernel3D<RAi_SIZE> RA; //At A1->A2, 58% occupancy for 16^3

	UNROLL_GENA1
		for (auto rax = 0; rax < RAi_SIZE; rax++) {
		
			UNROLL_GENA1
				for (auto ray = 0; ray < RAi_SIZE; ray++) {
					UNROLL_GENA1
						for (auto raz = 0; raz < RAi_SIZE; raz++) {
						

							int3 voxRA = NspaceIvox + make_int3(rax - RAi_HALF, ray - RAi_HALF, raz - RAi_HALF);
							if (!_isValidPos(resAprev, voxRA)) {
								RA.v[rax][ray][raz] = 0.0;
								continue;
							}


							//ivox + rx,y,z is in N space
							double sum = 0.0;
							for (auto rx = 0; rx < RESTR_SIZE; rx++) {
								for (auto rz = 0; rz < RESTR_SIZE; rz++) {
									for (auto ry = 0; ry < RESTR_SIZE; ry++) {									
										double rval = R.v[rx][ry][rz];
										int3 rvox = NspaceIvox + make_int3(rx - RESTR_HALF, ry - RESTR_HALF, rz - RESTR_HALF);
										if (!_isValidPos(resAprev, rvox)) {
											continue;
										}

										size_t NspaceIndex = _linearIndex(resAprev, rvox);
										
										{
											const MGGPU_Kernel3D<Ai_SIZE> & A = Aprev[NspaceIndex];
											int3 Aoffset = voxRA - rvox + make_int3(Ai_HALF);
											
											if (!_isValidPos(make_uint3(Ai_SIZE), Aoffset)) {
												continue;
											}

											double aval = A.v[Aoffset.x][Aoffset.y][Aoffset.z];
											sum += aval*rval;


										}

									}
								}
							}


							RA.v[rax][ray][raz] = sum;
						}
				}
		}




	size_t a1index = _linearIndex(resAnext, ivox);
	MGGPU_Kernel3D<Ai_SIZE> & aNext = Anext[a1index];


	for (auto xa1 = 0; xa1 < Ai_SIZE; xa1++) {	
		for (auto ya1 = 0; ya1 < Ai_SIZE; ya1++) {			
			for (auto za1 = 0; za1 < Ai_SIZE; za1++) {
				
				
				//ivox ... ivox / Cratio, Cratio = 1
				int3 voxA1 = ivox + make_int3(xa1 - Ai_HALF, ya1 - Ai_HALF, za1 - Ai_HALF);

				size_t debugJ = _linearIndex(resAnext, voxA1);

				if (!_isValidPos(resAprev, voxA1)) {
					aNext.v[xa1][ya1][za1] = 0.0;
					continue;
				}


				

				double sum = 0.0;
				for (auto rax = 0; rax < RAi_SIZE; rax++) {
					for (auto raz = 0; raz < RAi_SIZE; raz++) {
						for (auto ray = 0; ray < RAi_SIZE; ray++) {						

							
							int3 ravox = NspaceIvox + make_int3(rax - RAi_HALF, ray - RAi_HALF, raz - RAi_HALF);
							size_t NspaceIndex = _linearIndex(resAprev, ravox);

							if (!_isValidPos(resAprev, ravox)) {
								continue;
							}

							double rval = RA.v[rax][ray][raz];
							

							

							{

								int3 offset = voxA1 - make_int3(ravox.x / 2, ravox.y / 2, ravox.z / 2) + make_int3(INTERP_SIZE / 2);
								if (!_isValidPos(make_uint3(INTERP_SIZE), offset)) {
									continue;
								}

								double ival = I[NspaceIndex].v[offset.x][offset.y][offset.z];

								sum += ival*rval;							

							}
						}
					}
				}


				aNext.v[xa1][ya1][za1] = sum;

			}
		}
	}


	return;
}


__global__ void __buildA1(
	const uint3 resA0,
	const uint3 resA1,
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_InterpKernel * I,
	MGGPU_Kernel3D<5> * A1)
{

	//Get voxel position
	VOLUME_VOX_GUARD(resA1);
	int3 ivox = make_int3(vox);	

	buildA1At(
		ivox,
		resA0,
		resA1,
		A0,
		I,
		A1,
		const_sys_params.dirPrimary
	);	
}

__global__ void __buildAi(
	const uint3 resAprev,
	const uint3 resAnext,
	const MGGPU_Kernel3D<Ai_SIZE> * Aprev,
	const MGGPU_InterpKernel * I,
	MGGPU_Kernel3D<Ai_SIZE> * Anext)
{

	//Get voxel position
	VOLUME_VOX_GUARD(resAnext);
	int3 ivox = make_int3(vox);

	buildAiAt(
		ivox,
		resAprev,
		resAnext,
		Aprev,
		I,
		Anext,
		const_sys_params.dirPrimary
	);
}

bool MGGPU_BuildA1(
	const uint3 resA,
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_InterpKernel * I,
	MGGPU_Kernel3D<5> * A1,
	bool onDevice
) {

	int3 ivox;
	uint3 resA0 = resA;
	uint3 resA1 = make_uint3((resA0.x +1) / 2, (resA0.y + 1) / 2, (resA0.z + 1) / 2);

	if (onDevice) {		
		BLOCKS3D(A1_THREADS_PER_DIM, resA1); //TODO increase
		__buildA1 <<< numBlocks, block >> >(			
			resA0,
			resA1,
			A0,
			I,
			A1);
	}
	else {		
		for (ivox.z = 0; ivox.z < resA1.x; ivox.z++) {
			for (ivox.y = 0; ivox.y < resA1.y; ivox.y++) {
				for (ivox.x = 0; ivox.x < resA1.z; ivox.x++) {
					buildA1At(
						ivox,
						resA,
						resA1,
						A0,
						I,
						A1,
						const_sys_params_cpu.dirPrimary						
					);
				}
			}
		}
	}

	return true;

}

bool MGGPU_BuildAi(
	const uint3 resA,
	const MGGPU_Kernel3D<5> * Aprev,
	const MGGPU_InterpKernel * I,
	MGGPU_Kernel3D<5> * Anext,
	bool onDevice
) {

	int3 ivox;
	uint3 resAprev = resA;
	//uint3 resAnext = make_uint3(resAprev.x / 2, resAprev.y / 2, resAprev.z / 2);
	uint3 resAnext = make_uint3((resAprev.x +1) / 2, (resAprev.y + 1) / 2, (resAprev.z + 1) / 2);

	if (onDevice) {
		BLOCKS3D(Ai_THREADS_PER_DIM, resAnext); //TODO increase
		__buildAi << < numBlocks, block >> >(
			resAprev,
			resAnext,
			Aprev,
			I,
			Anext);
	}
	else {
		for (ivox.z = 0; ivox.z < resAnext.x; ivox.z++) {
			for (ivox.y = 0; ivox.y < resAnext.y; ivox.y++) {
				for (ivox.x = 0; ivox.x < resAnext.z; ivox.x++) {
					buildAiAt(
						ivox,
						resAprev,
						resAnext,
						Aprev,
						I,
						Anext,
						const_sys_params_cpu.dirPrimary
					);
				}
			}
		}
	}

	return true;

}



////////////////////////////////////////////// Residual

//Convolve by system top kernel
double __device__  convolve3D_SystemTop(
	const int3 & ivox,
	const MGGPU_SystemTopKernel & k,
	const MGGPU_Volume & x
	){

	//Assumes ivox is inside the boundary

	double sum = 0.0;

	if (ivox.z - 1 >= 0)
		sum += k.v[Z_NEG] * read<double>(x.surf, ivox + make_int3(0,0,-1));

	if (ivox.y - 1 >= 0)
		sum += k.v[Y_NEG] * read<double>(x.surf, ivox + make_int3(0, -1, 0));

	if (ivox.x - 1 >= 0)
		sum += k.v[X_NEG] * read<double>(x.surf, ivox + make_int3(-1, 0, 0));
	
	sum += k.v[DIR_NONE] * read<double>(x.surf, ivox);

	if (ivox.x + 1 < x.res.x)
		sum += k.v[X_POS] * read<double>(x.surf, ivox + make_int3(1, 0, 0));

	if (ivox.y + 1 < x.res.y)
		sum += k.v[Y_POS] * read<double>(x.surf, ivox + make_int3(0, 1, 0));

	if (ivox.z + 1 < x.res.z)
		sum += k.v[Z_POS] * read<double>(x.surf, ivox + make_int3(0, 0, 1));

	return sum;
}

template<size_t kN>
double __device__  convolve3D(
	const int3 & ivox,
	const MGGPU_Kernel3D<kN> & k,
	const MGGPU_Volume & vec
) {

	double sum = 0.0;

	const int kNHalf = MGGPU_KernelHalf(kN);
	const int3 offset = make_int3(-kNHalf);

	#pragma unroll
	for (int x = 0; x < kN; x++) {
		#pragma unroll
		for (int y = 0; y < kN; y++) {
			#pragma unroll
			for (int z = 0; z < kN; z++) {

				const int3 pos = ivox + make_int3(x, y, z) + offset;
				if (!_isValidPos(vec.res, pos))
					continue;
				
				sum += k.v[x][y][z] * read<double>(vec.surf, pos);
			}
		}
	}



	return sum;
}


__global__ void __residual_topLevel(
	const uint3 res,
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_Volume x,
	const MGGPU_Volume f,
	MGGPU_Volume r
) {
	VOLUME_IVOX_GUARD(res);	

	const size_t I = _linearIndex(res, ivox);

	const double Axval = convolve3D_SystemTop(ivox, A0[I], x);
	const double fval = read<double>(f.surf, ivox);
	const double rval = fval - Axval;

	write<double>(r.surf, ivox, rval);

}
__global__ void __residual(
	const uint3 res,
	const MGGPU_Kernel3D<Ai_SIZE> * A,
	const MGGPU_Volume x,
	const MGGPU_Volume f,
	MGGPU_Volume r
) {
	VOLUME_IVOX_GUARD(res);

	const size_t I = _linearIndex(res, ivox);
	const double Axval = convolve3D<Ai_SIZE>(ivox, A[I], x);
	const double fval = read<double>(f.surf, ivox);
	const double rval = fval - Axval;
	write<double>(r.surf, ivox, rval);

}



void MGGPU_Residual_TopLevel(
	const uint3 res,
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_Volume & x,
	const MGGPU_Volume & f,
	MGGPU_Volume & r
) {
	BLOCKS3D(8, res); 
	__residual_topLevel << < numBlocks, block >> >(	res,A0,x,f,	r);
}


void MGGPU_Residual(
	const uint3 res,
	const MGGPU_Kernel3D<Ai_SIZE> * A,
	const MGGPU_Volume & x,
	const MGGPU_Volume & f,
	MGGPU_Volume & r
) {
	BLOCKS3D(8, res);
	__residual << < numBlocks, block >> >(res, A, x, f, r);
}


////////////////////////////////////////////// Reductions

double MGGPU_SquareNorm(
	const uint3 res, 
	MGGPU_Volume & x,
	void * auxGPU, 
	void * auxCPU
	) {

	double result = 0.0;

	launchReduceKernel(
		TYPE_DOUBLE,
		REDUCE_OP_SQUARESUM,
		res,
		x.surf,
		auxGPU,
		auxCPU,
		&result
	);

	return result;

}

void MGGPU_SetToZero(
	MGGPU_Volume & x
) {
	double val = 0.0;
	launchClearKernel(TYPE_DOUBLE, x.surf, x.res, &val);
}


////////////////////////////////////////////// Smoothing



template <bool isTopLevel>
__global__ void __gaussSeidelZeroGuess( //assumes zero x guess
	uint3 res,
	MGGPU_Volume F,
	MGGPU_Volume X,
	MGGPU_KernelPtr A,
	double alpha
){
	VOLUME_IVOX_GUARD(res);

	const double fval = read<double>(F.surf, ivox);
	const size_t rowI = _linearIndex(res, ivox);
	double diag;
	if (isTopLevel) {
		diag = ((MGGPU_SystemTopKernel*)A)[rowI].v[DIR_NONE];
	}
	else {
		diag = ((const MGGPU_Kernel3D<Ai_SIZE>*)A)[rowI].v[Ai_HALF][Ai_HALF][Ai_HALF];
	}
	double newVal = fval / diag;

	double oldVal = read<double>(X.surf, ivox);
	newVal = alpha * newVal + (1.0 - alpha) * oldVal;

	write<double>(X.surf, ivox, newVal);
	
	

	

}


template <int dir, int sgn, bool alternate, bool isTopLevel>
__global__ void __gaussSeidelLine(
	uint3 res,
	MGGPU_Volume F,
	MGGPU_Volume X,
	MGGPU_KernelPtr A,
	double alpha
) {

	VOLUME_VOX; //no guard

	//Set directions
	const uint primDim = _at<uint>(res, dir); //16
	const int secDirs[2] = { 
		(dir + 1) % 3, // 2 half block
		(dir + 2) % 3 //3%3 = 0 full block
	};

	//Interleave second dir
	_at<uint>(vox, secDirs[0]) *= 2;

	//Alternate 0,1 start
	if (!alternate)
		_at<uint>(vox, secDirs[0]) += _at<uint>(vox, secDirs[1]) % 2;
	else
		_at<uint>(vox, secDirs[0]) += 1 - _at<uint>(vox, secDirs[1]) % 2;

	//Guard
	if (_at<uint>(vox, secDirs[0]) >= _at<uint>(res, secDirs[0]) ||
		_at<uint>(vox, secDirs[1]) >= _at<uint>(res, secDirs[1]))
		return;

	//Bounds
	const int begin = (sgn == -1) ? int(primDim) - 1 : 0;
	const int end = (sgn == -1) ? -1 : int(primDim);


	for (int i = begin; i != end; i += sgn) {
		int3 voxi = { vox.x, vox.y, vox.z };
		_at<int>(voxi, dir) = i;


		const size_t rowI = _linearIndex(res, voxi);

		double sum;
		double diag;

		if (isTopLevel) {
			MGGPU_SystemTopKernel a = ((MGGPU_SystemTopKernel*)A)[rowI];
			diag = a.v[DIR_NONE];
			a.v[DIR_NONE] = 0.0;
			sum = convolve3D_SystemTop(voxi, a, X);

		}
		else {
			//Avoid copy here?
			MGGPU_Kernel3D<Ai_SIZE> a = ((const MGGPU_Kernel3D<Ai_SIZE>*)A)[rowI];
			diag = a.v[Ai_HALF][Ai_HALF][Ai_HALF];
			a.v[Ai_HALF][Ai_HALF][Ai_HALF] = 0.0;
			sum = convolve3D<Ai_SIZE>(voxi, a, X);

		}

		

		double fval = read<double>(F.surf, voxi);
		double newVal = (fval - sum) / diag;		
		
		double oldVal = read<double>(X.surf, voxi);

		newVal = alpha * newVal + (1.0 - alpha) * oldVal;

		//printf("%d %f %f %f -> %f (<- %f) \n", i,sum, diag,fval, newVal, read<double>(X.surf, voxi));

		write<double>(X.surf, voxi, newVal);

		
	}


}

template<bool isTopLevel, int sgn>
void gaussSeidelX(MGGPU_SmootherParams & p, int blockDim) {
	uint3 block;
	uint3 numBlocks;

	block = make_uint3(1, blockDim, blockDim);
	numBlocks = make_uint3(
		1,
		((p.res.y / 2 + (block.y - 1)) / block.y),
		((p.res.z + (block.z - 1)) / block.z)
	);

	__gaussSeidelLine<0, sgn, false, isTopLevel> << <numBlocks, block >> > (p.res, p.f, p.x, p.A, p.alpha);
	__gaussSeidelLine<0, sgn, true, isTopLevel> << <numBlocks, block >> > (p.res, p.f, p.x, p.A, p.alpha);
}

template<bool isTopLevel, int sgn>
void gaussSeidelY(MGGPU_SmootherParams & p, int blockDim) {
	uint3 block;
	uint3 numBlocks;

	block = make_uint3(blockDim, 1, blockDim);
	numBlocks = make_uint3(
		((p.res.x + (block.x - 1)) / block.x),
		1,
		((p.res.z / 2 + (block.z - 1)) / block.z)
	);

	__gaussSeidelLine<1, sgn, false, isTopLevel> << <numBlocks, block >> > (p.res, p.f, p.x, p.A, p.alpha);
	__gaussSeidelLine<1, sgn, true, isTopLevel> << <numBlocks, block >> > (p.res, p.f, p.x, p.A, p.alpha);
}

template<bool isTopLevel, int sgn>
void gaussSeidelZ(MGGPU_SmootherParams & p, int blockDim) {
	uint3 block;
	uint3 numBlocks;

	block = make_uint3(blockDim, blockDim, 1);
	numBlocks = make_uint3(
		((p.res.x / 2 + (block.x - 1)) / block.y),
		((p.res.y + (block.y - 1)) / block.y),
		1
	);

	__gaussSeidelLine<2, sgn, false, isTopLevel> << <numBlocks, block >> > (p.res, p.f, p.x, p.A, p.alpha);
	__gaussSeidelLine<2, sgn, true, isTopLevel> << <numBlocks, block >> > (p.res, p.f, p.x, p.A, p.alpha);
}

template<bool isTopLevel>
void gaussSeidelAllDirs(MGGPU_SmootherParams & p, int blockDim) {

	if (p.dir == X_NEG) {
		gaussSeidelX<isTopLevel, 1>(p, blockDim);
		gaussSeidelY<isTopLevel, 1>(p, blockDim);
		gaussSeidelZ<isTopLevel, 1>(p, blockDim);
	}

	if (p.dir == Y_NEG) {
		gaussSeidelY<isTopLevel, 1>(p, blockDim);
		gaussSeidelZ<isTopLevel, 1>(p, blockDim);
		gaussSeidelX<isTopLevel, 1>(p, blockDim);
	}
	

	if (p.dir == Z_NEG) {
		gaussSeidelZ<isTopLevel, 1>(p, blockDim);
		gaussSeidelX<isTopLevel, 1>(p, blockDim);
		gaussSeidelY<isTopLevel, 1>(p, blockDim);
	}

	if (p.dir == X_POS) {
		gaussSeidelX<isTopLevel, -1>(p, blockDim);
		gaussSeidelY<isTopLevel, -1>(p, blockDim);
		gaussSeidelZ<isTopLevel, -1>(p, blockDim);
	}

	if (p.dir == Y_POS) {
		gaussSeidelY<isTopLevel, -1>(p, blockDim);
		gaussSeidelZ<isTopLevel, -1>(p, blockDim);
		gaussSeidelX<isTopLevel, -1>(p, blockDim);
	}


	if (p.dir == Z_POS) {
		gaussSeidelZ<isTopLevel, -1>(p, blockDim);
		gaussSeidelX<isTopLevel, -1>(p, blockDim);
		gaussSeidelY<isTopLevel, -1>(p, blockDim);
	}



	

	/*if (!p.isTopLevel) {
		BLOCKS3D(8, p.res);
		__gaussSeidelZeroGuess<false><<<numBlocks, block>>>(p.res, p.f, p.x, p.A, 0.5);
	}*/

	
	
	


	
}

//#define GS_CHECK_RESIDUAL
double MGGPU_GaussSeidel(MGGPU_SmootherParams & p) {


	double error = 0.0;
#ifdef GS_CHECK_RESIDUAL
	
	double fsq = MGGPU_SquareNorm(p.res, p.f, p.auxBufferGPU, p.auxBufferCPU);

	if (fsq == 0.0) {
		printf("WARNING ||f||^2 = 0.0\n");
	}	
#endif

	for (auto i = 0; i < p.iter; ++i) {


		if (p.isTopLevel) {
			gaussSeidelAllDirs<true>(p, 8);
#ifdef GS_CHECK_RESIDUAL
			MGGPU_Residual_TopLevel(p.res, (MGGPU_SystemTopKernel *)p.A, p.x, p.f, p.r);
#endif
		}
		else {
			gaussSeidelAllDirs<false>(p, 8);
#ifdef GS_CHECK_RESIDUAL
			MGGPU_Residual(p.res, (MGGPU_Kernel3D<Ai_SIZE>*)p.A, p.x, p.f, p.r);
#endif

		}

		//printf("i %d\n", i);

#ifdef GS_CHECK_RESIDUAL
		//Might not be needed to calculate residual
		double rsq = MGGPU_SquareNorm(p.res, p.r, p.auxBufferGPU, p.auxBufferCPU);		
		
		error = sqrt(rsq / fsq);

		//printf("rsq: %f, fsq: %f, error: %f\n", rsq,fsq,error);

		if (error <= p.tolerance)
			break;
#endif

	}


#ifndef GS_CHECK_RESIDUAL
	if (p.isTopLevel) {
		MGGPU_Residual_TopLevel(p.res, (MGGPU_SystemTopKernel *)p.A, p.x, p.f, p.r);
	}
	else {
		MGGPU_Residual(p.res, (MGGPU_Kernel3D<Ai_SIZE>*)p.A, p.x, p.f, p.r);
	}
#endif

	return error;
}



template <bool isTopLevel>
__global__ void __jacobiKernel(
	uint3 res,
	MGGPU_Volume F,
	MGGPU_Volume X,
	MGGPU_Volume Y,
	MGGPU_KernelPtr A
) {	
	VOLUME_IVOX_GUARD(res);

	const size_t rowI = _linearIndex(res, ivox);

	double sum;
	double diag;

	if (isTopLevel) {
		MGGPU_SystemTopKernel a = ((MGGPU_SystemTopKernel*)A)[rowI];
		diag = a.v[DIR_NONE];
		a.v[DIR_NONE] = 0.0;
		sum = convolve3D_SystemTop(ivox, a, X);

	}
	else {
		//Avoid copy here?
		MGGPU_Kernel3D<Ai_SIZE> a = ((const MGGPU_Kernel3D<Ai_SIZE>*)A)[rowI];
		diag = a.v[Ai_HALF][Ai_HALF][Ai_HALF];
		a.v[Ai_HALF][Ai_HALF][Ai_HALF] = 0.0;
		sum = convolve3D<Ai_SIZE>(ivox, a, X);

	}

	double fval = read<double>(F.surf, ivox);
	double newVal = (fval - sum) / diag;

	write<double>(Y.surf, ivox, newVal);	


}


void MGGPU_Jacobi(MGGPU_SmootherParams & p) {

	MGGPU_Volume src, dest;
	src = p.x;
	dest = p.tmpx;

	const int iter = (p.iter % 2 == 0) ? p.iter : p.iter + 1;
		

	for (auto i = 0; i < iter; ++i) {
		BLOCKS3D(4, p.res);
		if (p.isTopLevel) {
			const bool top = true;
			__jacobiKernel<top> << <numBlocks, block >>> (
				p.res,
				p.f,
				src,
				dest,
				p.A
				);
		}
		else {
			const bool top = false;
			__jacobiKernel<top> << <numBlocks, block >> > (
				p.res,
				p.f,
				src,
				dest,
				p.A
				);
		}
		//Swap 
		MGGPU_Volume tmp;
		tmp = src;
		src = dest;
		dest = tmp;
	}

	//Calculate residual at the end
	//only for restr, todo: move out
	if (p.isTopLevel) {
		MGGPU_Residual_TopLevel(p.res, (MGGPU_SystemTopKernel *)p.A, p.x, p.f, p.r);
	}
	else {
		MGGPU_Residual(p.res, (MGGPU_Kernel3D<Ai_SIZE>*)p.A, p.x, p.f, p.r);
	}

}



__global__ void ___restrictVolume(
	MGGPU_Volume xPrev, MGGPU_Volume xNext
){	
	VOLUME_IVOX_GUARD(xNext.res);
	const MGGPU_RestrictKernel R = MGGPU_GetRestrictionKernel(make_uint3(ivox), xNext.res, const_sys_params.dirPrimary);
	int3 ivoxPrev = make_int3(ivox.x * 2, ivox.y * 2, ivox.z * 2);
	const double Rxval = convolve3D<RESTR_SIZE>(ivoxPrev, R, xPrev);
	write<double>(xNext.surf, ivox, Rxval);
}

//Requires const_sys_params
void MGGPU_Restrict(
	MGGPU_Volume & xPrev,
	MGGPU_Volume & xNext
) {

	BLOCKS3D(4, xNext.res);
	___restrictVolume << < numBlocks, block >> >(xPrev, xNext);

}


__global__ void ___interpVolumeAndAdd(
	MGGPU_Volume xPrev, 
	MGGPU_Volume xNext,
	MGGPU_InterpKernel * I
) {
	VOLUME_IVOX_GUARD(xNext.res);

	const size_t linIndex = _linearIndex(xNext.res, ivox);	

	int3 ivoxPrev = make_int3(ivox.x / 2, ivox.y / 2, ivox.z / 2);
	const MGGPU_InterpKernel & kern = I[linIndex];
	const double Ixval = convolve3D<INTERP_SIZE>(ivoxPrev, kern, xPrev);
		
	//read
	double oldVal = read<double>(xNext.surf, ivox);	
	double newVal = oldVal + Ixval;	
	write<double>(xNext.surf, ivox, newVal);
}

//Requires const_sys_params
void MGGPU_InterpolateAndAdd(
	MGGPU_Volume & xPrev,
	MGGPU_Volume & xNext,
	MGGPU_InterpKernel * I
) {

	BLOCKS3D(4, xNext.res);
	___interpVolumeAndAdd << < numBlocks, block >> >(xPrev, xNext, I);

}



#ifdef ____OLD


__global__ void __genA1(
	const uint3 Nres,
	const uint3 Nhalfres,
	const MGGPU_Kernel3D<4> * IT0,
	MGGPU_Kernel3D<5> * output
) {
	VOLUME_VOX_GUARD(Nhalfres);


	/*int3 ivox = make_int3(vox);
	//int3 ivoxHalf = make_int3(vox.x / 2, vox.y / 2, vox.z / 2);

	size_t i = _linearIndex(Nhalfres, ivox);

	MGGPU_RestrictKernel R = MGGPU_GetRestrictionKernel(vox, Nhalfres, const_sys_params.dirPrimary);

	for (auto x_r = 0; x_r < RESTR_SIZE; x_r++) {
	for (auto y_r = 0; y_r < RESTR_SIZE; y_r++) {
	for (auto z_r = 0; z_r < RESTR_SIZE; z_r++) {

	int3 kVox = 2*ivox + make_int3(x_r, y_r, z_r) - make_int3(RESTR_SIZE / 2 - 1);
	if (!_isValidPos(Nres, kVox))
	continue;

	size_t ki = _linearIndex(kVox, Nres);

	}
	}
	}

	*/


}


void MGGPU_GenerateA1(
	const uint3 & Nres,
	const uint3 & Nhalfres,
	const MGGPU_Kernel3D<4> * IT0,
	MGGPU_Kernel3D<5> * output
) {

	BLOCKS3D(2, Nhalfres);
	__genA1 << < numBlocks, block >> > (
		Nres,
		Nhalfres,
		IT0,
		output
		);

}



void MGGPU_GenerateTranposeInterpKernels(
	const uint3 & Nres,
	const uint3 & Nhalfres,
	const MGGPU_Volume & domainHalf,
	MGGPU_Kernel3D<4> * output
) {


	BLOCKS3D(2, Nres);
	___genITranpose << < numBlocks, block >> > (
		Nres, Nhalfres,
		domainHalf,
		output
		);

}

__global__ void ___convolve_A0_IT0_Direct(
	const uint3 Nres,
	const uint3 Nhalfres,
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_Kernel3D<4> * IT0,
	MGGPU_Kernel3D<3> * output
) {

	VOLUME_VOX_GUARD(Nres);

	size_t iN = _linearIndex(Nres, vox);
	size_t iNHalf = _linearIndex(Nhalfres, vox);


	int3 voxi = make_int3(vox);
	int3 voxiHalf = make_int3(voxi.x / 2, voxi.y / 2, voxi.z / 2);

	const int N_A = 3;
	const int N_I = 4;
	const int STRIDE = 2;
	const int N_AI = (N_A + N_I - 1) / STRIDE; //3	


											   //Read packed a0 kernel
	size_t i = _linearIndex(Nres, vox);
	MGGPU_Kernel3D<N_AI> & AI = output[i];

	const MGGPU_SystemTopKernel & a7 = A0[i];

	//Scatter to 3x3x3 kernel
	MGGPU_Kernel3D<3> a;
	{
		memset(&a, 0, sizeof(MGGPU_Kernel3D<3>));
		a.v[1][1][1] = a7.v[DIR_NONE];
		a.v[0][1][1] = a7.v[X_NEG];
		a.v[2][1][1] = a7.v[X_POS];
		a.v[1][0][1] = a7.v[Y_NEG];
		a.v[1][2][1] = a7.v[Y_POS];
		a.v[1][1][0] = a7.v[Z_NEG];
		a.v[1][1][2] = a7.v[Z_POS];
	}


	for (int x_ai = 0; x_ai < N_AI; x_ai++) {
		for (int y_ai = 0; y_ai < N_AI; y_ai++) {
			for (int z_ai = 0; z_ai < N_AI; z_ai++) {

				int3 offsetAICenter = make_int3(-N_AI / 2) + make_int3(x_ai, y_ai, z_ai);
				int3 interpPos = voxiHalf + offsetAICenter;

				if (!_isValidPos(Nhalfres, interpPos)) {
					AI.v[x_ai][y_ai][z_ai] = 0.0;
					continue;
				}

				size_t interpIndex = _linearIndex(Nhalfres, voxiHalf + offsetAICenter);

				const MGGPU_Kernel3D<N_I> & I = IT0[interpIndex];

				double sum = 0.0;
				//dot with offseted a
				for (int x_i = 0; x_i < N_I; x_i++) {
					for (int y_i = 0; y_i < N_I; y_i++) {
						for (int z_i = 0; z_i < N_I; z_i++) {




							int3 offsetICenter = make_int3(x_i, y_i, z_i) - make_int3(N_I / 2 - 1);
							int3 voxI = 2 * (interpPos)+offsetICenter;


							int3 apos = voxI - voxi + make_int3(1, 1, 1);


							int x_a = apos.x;
							int y_a = apos.y;
							int z_a = apos.z;

							if (!_isValidPos(make_uint3(N_A), make_int3(x_a, y_a, z_a)))
								continue;

							sum += I.v[x_i][y_i][z_i] * a.v[x_a][y_a][z_a];
						}
					}
				}

				AI.v[x_ai][y_ai][z_ai] = sum;
			}
		}
	}


}



void MGGPU_GenerateAI0(
	const uint3 & Nres,
	const uint3 & Nhalfres,
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_Kernel3D<4> * IT0,
	MGGPU_Kernel3D<3> * output
) {

	BLOCKS3D(2, Nres);
	___convolve_A0_IT0_Direct << < numBlocks, block >> > (
		Nres,
		Nhalfres,
		A0,
		IT0,
		output
		);

}



void __device__ MGGPU_Convolve_A0_I_Direct(
	const uint3 destRes,
	const MGGPU_Volume & domain,
	const MGGPU_SystemTopKernel * A0,
	const uint3 & vox,
	int dirIndex,
	MGGPU_Kernel3D<5> * out
) {
	const int3 voxi = make_int3(vox);

	const int N_A = 3;
	const int N_I = 3;
	const int N_AI = N_A + N_I - 1; //5

	MGGPU_Kernel3D<N_AI> & AI = *out;


	//Read packed a0 kernel
	size_t i = _linearIndex(destRes, vox);
	const MGGPU_SystemTopKernel & a7 = A0[i];

	//Scatter to 3x3x3 kernel
	MGGPU_Kernel3D<3> a;
	{
		memset(&a, 0, sizeof(MGGPU_Kernel3D<3>));
		a.v[1][1][1] = a7.v[DIR_NONE];
		a.v[0][1][1] = a7.v[X_NEG];
		a.v[2][1][1] = a7.v[X_POS];
		a.v[1][0][1] = a7.v[Y_NEG];
		a.v[1][2][1] = a7.v[Y_POS];
		a.v[1][1][0] = a7.v[Z_NEG];
		a.v[1][1][2] = a7.v[Z_POS];
	}



	for (int x_ai = 0; x_ai < N_AI; x_ai++) {
		for (int y_ai = 0; y_ai < N_AI; y_ai++) {
			for (int z_ai = 0; z_ai < N_AI; z_ai++) {
				int3 offsetACenter = make_int3(-N_AI / 2) + make_int3(x_ai, y_ai, z_ai);


				//Get I kernel at _ai pos
				MGGPU_InterpKernel I = MGGPU_GetInterpolationKernel(domain, voxi + offsetACenter, destRes, dirIndex);

				double sum = 0.0;

				//dot with offseted a
				for (int x_i = 0; x_i < N_I; x_i++) {
					for (int y_i = 0; y_i < N_I; y_i++) {
						for (int z_i = 0; z_i < N_I; z_i++) {

							int3 offsetICenter = make_int3(-N_I / 2) + make_int3(x_i, y_i, z_i);

							int3 apos = offsetACenter + make_int3(N_A / 2) + offsetICenter;
							int x_a = apos.x;
							int y_a = apos.y;
							int z_a = apos.z;



							if (!_isValidPos(make_uint3(N_A), make_int3(x_a, y_a, z_a)))
								continue;

							sum += I.v[x_i][y_i][z_i] * a.v[x_a][y_a][z_a];
						}
					}
				}

				AI.v[x_ai][y_ai][z_ai] = sum;
			}
		}
	}


}



__global__ void ___convolve_A0_I_Direct(
	const uint3 destRes,
	const MGGPU_Volume domainHalf,
	const MGGPU_SystemTopKernel * A0,
	MGGPU_Kernel3D<5> * AI
) {

	VOLUME_VOX_GUARD(destRes);

	size_t i = _linearIndex(destRes, vox);

	MGGPU_Convolve_A0_I_Direct(destRes, domainHalf, A0, vox, const_sys_params.dirPrimary, &(AI[i]));


}

void MGGPU_GenerateAI0(
	const MGGPU_Volume & domainHalf,
	const MGGPU_SystemTopKernel * A0,
	MGGPU_Kernel3D<5> * AI
) {

	const uint3 destRes = domainHalf.res * 2; // TODO:

	BLOCKS3D(2, destRes);
	___convolve_A0_I_Direct << < numBlocks, block >> > (
		destRes,
		domainHalf,
		A0,
		AI
		);

}

#endif