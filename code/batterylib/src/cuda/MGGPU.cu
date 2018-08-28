#include "MGGPU.cuh"
#include <stdio.h>

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


//Kernel for 3D convolution by 2^3 kernel with stride 2 and border
__global__ void ___convolve3D_2_2(
	const MGGPU_Volume in,	
	const MGGPU_Volume out
){
	//todo shared mem
	VOLUME_VOX_GUARD(out.res);

	double sum = 0.0;

	MGGPU_Kernel3D<2> & k = *((MGGPU_Kernel3D<2>*)const_kernel);
	
	//TODO: boundary handling (upper only)
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

}


/*
	Kernel generation
*/

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



__device__  MGGPU_InterpKernel MGGPU_GetInterpolationKernel(
	const MGGPU_Volume & domainSrc,
	const int3 & vox, //vox in destination
	const uint3 & destRes, //should be exactly double (if power of 2) of domain.res
	int dirIndex
) {

	/*
	Two spaces:
	source : n/2 (domain, domain.res)
	dest: n (vox, destRes)
	*/

	MGGPU_InterpKernel kernel;
	
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
			w[i] *= read<double>(domainSrc.surf, make_uint3(voxSrcNew)); //redundant conversion to uint, TODO better
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
			if(!_isValidPos(domainSrc.res, voxSrc + offset)) {
				int3 p = voxSrc + offset;
				printf("%d %d %d\n", p.x, p.y, p.z);
			}
#endif

			//Read weight from source domain
			w[i] *= read<double>(domainSrc.surf, make_uint3(voxSrcNew));
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
	){

	VOLUME_VOX_GUARD(destRes);

	size_t i = _linearIndex(destRes, vox);

	MGGPU_Convolve_A0_I_Direct(destRes, domainHalf, A0, vox, const_sys_params.dirPrimary, &(AI[i]) );


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
	BLOCKS3D(2, destRes);
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



void MGGPU_GenerateTranposeInterpKernels(
	const uint3 & Nres,
	const uint3 & Nhalfres,
	const MGGPU_Volume & domainHalf,
	MGGPU_Kernel3D<4> * output
) {

	const uint totalOut = Nhalfres.x * Nhalfres.y * Nhalfres.z;
	

	BLOCKS3D(2, Nres);
	___genITranpose << < numBlocks, block >> > (
		Nres, Nhalfres,
		domainHalf,
		output
		);

}