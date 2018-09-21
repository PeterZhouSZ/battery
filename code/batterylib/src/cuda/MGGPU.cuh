#pragma once

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>

#include "Volume.cuh"
#include "MGGPU_Types.cuh"
#include "LinearSys.cuh"



bool MGGPU_commitSysParams(const LinearSys_SysParams & sysparams);



inline  __device__ __host__ int MGGPU_outputKernelSize(
	int Adim,
	int Bdim,
	int Bratio
) {
	int BdimTranpose = Bdim * Bratio;
	return (Adim + BdimTranpose - 1) / Bratio;
}

 
inline __device__ __host__ double MGGPU_GetTopLevelValue(const CUDA_Stencil_7 & k, const int3 & v) {


	if (v.x == -1) {
		if (v.y == 0 && v.z == 0)
			return k.v[X_NEG];
		return 0.0;
	}

	if (v.x == 0) {
		if (v.y == -1) {
			if (v.z == 0) return k.v[Y_NEG];
			return 0.0;
		}
		if (v.z == -1) {
			if (v.y == 0) return k.v[Z_NEG];
			return 0.0;
		}

		if (v.y == 1) {
			if (v.z == 0) return k.v[Y_POS];
			return 0.0;
		}

		if (v.z == 1) {
			if (v.y == 0) return k.v[Z_POS];
			return 0.0;
		}

		if (v.y == 0 && v.z == 0) {
			return k.v[DIR_NONE];
		}

		return 0.0;
	}


	if (v.x == 1) {
		if (v.y == 0 && v.z == 0)
			return k.v[X_POS];		
	}

	return 0.0;
}


inline __device__ __host__ MGGPU_DomainRestrictKernel MGGPU_GetDomainRestrictionKernel(){
	MGGPU_DomainRestrictKernel kernel;
	double v = 1.0 / 8.0;
	kernel.v[0][0][0] = v;
	kernel.v[0][0][1] = v;
	kernel.v[0][1][0] = v;
	kernel.v[0][1][1] = v;
	kernel.v[1][0][0] = v;
	kernel.v[1][0][1] = v;
	kernel.v[1][1][0] = v;
	kernel.v[1][1][1] = v;
	return kernel;
}

//Inplace version
inline __device__ __host__ void MGGPU_GetRestrictionKernel(
	const uint3 & vox,
	const uint3 & targetRes,
	int dirIndex,
	double * out
) {

	MGGPU_RestrictKernel & kern = *((MGGPU_RestrictKernel*)out);

	#pragma unroll
	for (int i = 0; i < 4; i += 3) {
		kern.v[i][0][0] = 1;
		kern.v[i][0][1] = 3;
		kern.v[i][0][2] = 3;
		kern.v[i][0][3] = 1;

		kern.v[i][1][0] = 3;
		kern.v[i][1][1] = 9;
		kern.v[i][1][2] = 9;
		kern.v[i][1][3] = 3;

		kern.v[i][2][0] = 3;
		kern.v[i][2][1] = 9;
		kern.v[i][2][2] = 9;
		kern.v[i][2][3] = 3;

		kern.v[i][3][0] = 1;
		kern.v[i][3][1] = 3;
		kern.v[i][3][2] = 3;
		kern.v[i][3][3] = 1;
	}

	#pragma unroll
	for (int i = 1; i<=2 ; i += 1) {
		kern.v[i][0][0] = 3;
		kern.v[i][0][1] = 9;
		kern.v[i][0][2] = 9;
		kern.v[i][0][3] = 3;

		kern.v[i][1][0] = 9;
		kern.v[i][1][1] = 27;
		kern.v[i][1][2] = 27;
		kern.v[i][1][3] = 9;

		kern.v[i][2][0] = 9;
		kern.v[i][2][1] = 27;
		kern.v[i][2][2] = 27;
		kern.v[i][2][3] = 9;

		kern.v[i][3][0] = 3;
		kern.v[i][3][1] = 9;
		kern.v[i][3][2] = 9;
		kern.v[i][3][3] = 3;
	}


	if (vox.x == 0 && dirIndex != 0) {
		for (auto j = 0; j < 4; j++) {
			for (auto k = 0; k < 4; k++) {
				kern.v[1][j][k] += kern.v[0][j][k];
				kern.v[0][j][k] = 0;
			}
		}
	}

	if (vox.x == targetRes.x - 1 && dirIndex != 0) {
		for (auto j = 0; j < 4; j++) {
			for (auto k = 0; k < 4; k++) {
				kern.v[2][j][k] += kern.v[3][j][k];
				kern.v[3][j][k] = 0;
			}
		}
	}

	if (vox.y == 0 && dirIndex != 1) {
		for (auto i = 0; i < 4; i++) {
			for (auto k = 0; k < 4; k++) {
				kern.v[i][1][k] += kern.v[i][0][k];
				kern.v[i][0][k] = 0;
			}
		}
	}

	if (vox.y == targetRes.y - 1 && dirIndex != 1) {
		for (auto i = 0; i < 4; i++) {
			for (auto k = 0; k < 4; k++) {
				kern.v[i][2][k] += kern.v[i][3][k];
				kern.v[i][3][k] = 0;
			}
		}
	}


	if (vox.z == 0 && dirIndex != 2) {
		for (auto i = 0; i < 4; i++) {
			for (auto j = 0; j < 4; j++) {
				kern.v[i][j][1] += kern.v[i][j][0];
				kern.v[i][j][0] = 0;
			}
		}
	}

	if (vox.z == targetRes.z - 1 && dirIndex != 2) {
		for (auto i = 0; i < 4; i++) {
			for (auto j = 0; j < 4; j++) {
				kern.v[i][j][2] += kern.v[i][j][3];
				kern.v[i][j][3] = 0;
			}
		}
	}

	double W = 0.0;
	for (auto i = 0; i < 4; i++) {
		for (auto j = 0; j < 4; j++) {
			for (auto k = 0; k < 4; k++) {
				W += kern.v[i][j][k];
			}
		}
	}

	for (auto i = 0; i < 4; i++) {
		for (auto j = 0; j < 4; j++) {
			for (auto k = 0; k < 4; k++) {
				kern.v[i][j][k] /= W;
			}
		}
	}	
}


inline __device__ __host__ MGGPU_RestrictKernel MGGPU_GetRestrictionKernel(
	const uint3 & vox, 
	const uint3 & targetRes,
	int dirIndex
) {

	MGGPU_RestrictKernel kernel;	

	double w[4][4][4] = {
		{
			{ 1,3,3,1 },
			{ 3,9,9,3 },
			{ 3,9,9,3 },
			{ 1,3,3,1 }
		},
		{
			{ 3,9,9,3 },
			{ 9,27,27,9 },
			{ 9,27,27,9 },
			{ 3,9,9,3 }
		},
		{
			{ 3,9,9,3 },
			{ 9,27,27,9 },
			{ 9,27,27,9 },
			{ 3,9,9,3 }
		},
		{
			{ 1,3,3,1 },
			{ 3,9,9,3 },
			{ 3,9,9,3 },
			{ 1,3,3,1 }
		},

	};

	if (vox.x == 0 && dirIndex != 0) {
		for (auto j = 0; j < 4; j++) {
			for (auto k = 0; k < 4; k++) {
				w[1][j][k] += w[0][j][k];
				w[0][j][k] = 0;
			}
		}
	}

	if (vox.x == targetRes.x - 1 && dirIndex != 0) {
		for (auto j = 0; j < 4; j++) {
			for (auto k = 0; k < 4; k++) {
				w[2][j][k] += w[3][j][k];
				w[3][j][k] = 0;
			}
		}
	}

	if (vox.y == 0 && dirIndex != 1) {
		for (auto i = 0; i < 4; i++) {
			for (auto k = 0; k < 4; k++) {
				w[i][1][k] += w[i][0][k];
				w[i][0][k] = 0;
			}
		}
	}

	if (vox.y == targetRes.y - 1 && dirIndex != 1) {
		for (auto i = 0; i < 4; i++) {
			for (auto k = 0; k < 4; k++) {
				w[i][2][k] += w[i][3][k];
				w[i][3][k] = 0;
			}
		}
	}


	if (vox.z == 0 && dirIndex != 2) {
		for (auto i = 0; i < 4; i++) {
			for (auto j = 0; j < 4; j++) {
				w[i][j][1] += w[i][j][0];
				w[i][j][0] = 0;
			}
		}
	}

	if (vox.z == targetRes.z - 1 && dirIndex != 2) {
		for (auto i = 0; i < 4; i++) {
			for (auto j = 0; j < 4; j++) {
				w[i][j][2] += w[i][j][3];
				w[i][j][3] = 0;
			}
		}
	}

	double W = 0.0;
	for (auto i = 0; i < 4; i++) {
		for (auto j = 0; j < 4; j++) {
			for (auto k = 0; k < 4; k++) {
				W += w[i][j][k];
			}
		}
	}

	for (auto i = 0; i < 4; i++) {
		for (auto j = 0; j < 4; j++) {
			for (auto k = 0; k < 4; k++) {
				w[i][j][k] /= W;
			}
		}
	}
	
	memcpy(&kernel, w, RESTR_SIZE * RESTR_SIZE * RESTR_SIZE * sizeof(double));
	return kernel;
}


inline __device__ __host__ int MGGPU_KernelProductSize(int an, int bn) {
	return an + bn - 1;
}


inline __device__ __host__ int MGGPU_KernelHalf(int kN) {
	if (kN % 2 == 0) {
		return kN / 2 - 1;
	}	
	return kN / 2;
}




/*
	Convolves using a single kernel (suports only 2^3 kernel, stride 2 at the moment)
*/
void MGGPU_Convolve(
	const CUDA_Volume & in,
	CUDA_KernelPtrD kernel, int kn,
	const CUDA_Volume & out
);


/*
Preparation
*/



/*
	Generates interpolation kernels
*/
void MGGPU_GenerateSystemInterpKernels(
	const uint3 & destRes,
	const CUDA_Volume & domain,
	MGGPU_InterpKernel * I
);


/*
	Builds A1 using Galerkin coarsening operator (special case of R0*A0*I0)
*/
bool MGGPU_BuildA1(
	const uint3 resA,
	const CUDA_Stencil_7 * A0,
	const MGGPU_InterpKernel * I,
	CUDA_Kernel3DD<5> * A1,
	bool onDevice
);

/*
	Builds Ai using Galerkin coarsening operator (Ri-1*Ai-1*Ii-1)
*/
bool MGGPU_BuildAi(
	const uint3 resA,
	const CUDA_Kernel3DD<5> * Aprev,
	const MGGPU_InterpKernel * I,
	CUDA_Kernel3DD<5> * Anext,
	bool onDevice
);




void MGGPU_Residual(
	const uint3 res,
	const CUDA_Kernel3DD<5> * A,
	const CUDA_Volume & x,
	const CUDA_Volume & f,
	CUDA_Volume & r
);




double MGGPU_GaussSeidel(MGGPU_SmootherParams & p);

void MGGPU_Jacobi(MGGPU_SmootherParams & p);



void MGGPU_Restrict(
	CUDA_Volume & xPrev,
	CUDA_Volume & xNext
);


void MGGPU_InterpolateAndAdd(
	CUDA_Volume & xPrev,
	CUDA_Volume & xNext,
	MGGPU_InterpKernel * I
);






#ifdef ___OLD


inline __device__ __host__ void MGGPU_A0(MGGPU_RestrictKernel & R, CUDA_Stencil_7 & A, MGGPU_InterpKernel & I) {

	//R*A


}

inline __device__ __host__ void MGGPU_CombineKernels(
	CUDA_KernelPtrD * a, int an,
	CUDA_KernelPtrD * b, int bn,
	int stride,
	CUDA_KernelPtrD * c
) {
	int cn = MGGPU_KernelProductSize(an, bn);


	/*
	a0 needs to be transformed from [7] -> 3x3x3
	a1 = r * a0 * i
	= 4^3 * 3^3 * 3^3
	c .. a1
	*/

}



void MGGPU_GenerateTranposeInterpKernels(
	const uint3 & Nres,
	const uint3 & Nhalfres,
	const CUDA_Volume & domainHalf,
	CUDA_Kernel3DD<4> * output
);

void MGGPU_GenerateAI0(
	const uint3 & Nres,
	const uint3 & Nhalfres,	
	const CUDA_Stencil_7 * A0,
	const CUDA_Kernel3DD<4> * IT0,
	CUDA_Kernel3DD<3> * output
);




bool MGGPU_CombineKernelsGeneric(
	const uint3 resArow,
	const uint3 resAcol,
	const uint3 resBrow,
	const uint3 resBcol,
	const CUDA_KernelPtrD A,
	const int Adim,
	const CUDA_KernelPtrD B,
	const int Bdim,
	CUDA_KernelPtrD C,
	const int Cdim,
	bool onDevice = true
);


bool MGGPU_CombineKernelsTopLevel(
	const uint3 resA,
	const uint3 resBrow,
	const uint3 resBcol,
	const CUDA_KernelPtrD A,
	const CUDA_KernelPtrD B,
	const int Bdim,
	CUDA_KernelPtrD C,
	const int Cdim,
	CUDA_Volume interpDomain,
	bool onDevice = true
);

bool MGGPU_CombineKernelsRestrict(
	const uint3 resArow,
	const uint3 resAcol,
	const uint3 resBrow,
	const uint3 resBcol,	
	const CUDA_KernelPtrD B,
	const int Bdim,
	CUDA_KernelPtrD C,
	bool onDevice = true
);

#endif