#pragma once

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>

#include "Volume.cuh"


struct MGGPU_Volume{
	uint3 res;	
	PrimitiveType type;
	cudaSurfaceObject_t surf;	
	int volID;

	void * cpu; //for debug
};


#define MAX_LEVELS 10

//Construction params
struct MGGPU_SysParams {	
	double highConc;
	double lowConc;
	double concetrationBegin;
	double concetrationEnd;
	double cellDim[3];
	double faceArea[3];
	int dirPrimary;
	uint2 dirSecondary;
	Dir dir;
};

bool commitSysParams(const MGGPU_SysParams & sysparams);

template <size_t size>
struct MGGPU_Kernel3D {
	double v[size][size][size];
};

//7-point stencil
struct MY_ALIGN(16) MGGPU_SystemTopKernel {
	double v[7];
};

#define INTERP_SIZE 3
#define RESTR_SIZE 4
#define DOMAIN_RESTR_SIZE 2

using MGGPU_InterpKernel = MGGPU_Kernel3D<INTERP_SIZE>;
using MGGPU_RestrictKernel = MGGPU_Kernel3D<RESTR_SIZE>;
using MGGPU_DomainRestrictKernel = MGGPU_Kernel3D<DOMAIN_RESTR_SIZE>;
using MGGPU_KernelPtr = double *;

inline  __device__ __host__ int MGGPU_outputKernelSize(
	int Adim,
	int Bdim,
	int Bratio
) {
	int BdimTranpose = Bdim * Bratio;
	return (Adim + BdimTranpose - 1) / Bratio;
}

 
inline __device__ __host__ double MGGPU_GetTopLevelValue(const MGGPU_SystemTopKernel & k, const int3 & v) {


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



/*
	Generates domain from mask and two double values
*/
void MGGPU_GenerateDomain(
	const MGGPU_Volume & binaryMask,
	double value_zero,
	double value_one,
	MGGPU_Volume & output
);


/*
	Convolves using a single kernel (suports only 2^3 kernel, stride 2 at the moment)
*/
void MGGPU_Convolve(
	const MGGPU_Volume & in,
	MGGPU_KernelPtr kernel, int kn,
	const MGGPU_Volume & out
);


/*
Preparation
*/


/*
	Generates 7-point stencil of linear system as a function of the domain
*/
void MGGPU_GenerateSystemTopKernel(
	const MGGPU_Volume & domain,
	MGGPU_SystemTopKernel * A0,
	MGGPU_Volume & f //rhs of lin sys
);

/*
	Generates interpolation kernels
*/
void MGGPU_GenerateSystemInterpKernels(
	const uint3 & destRes,
	const MGGPU_Volume & domain,
	MGGPU_InterpKernel * I
);


/*
	Builds A1 using Galerkin coarsening operator (special case of R0*A0*I0)
*/
bool MGGPU_BuildA1(
	const uint3 resA,
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_InterpKernel * I,
	MGGPU_Kernel3D<5> * A1,
	bool onDevice
);

/*
	Builds Ai using Galerkin coarsening operator (Ri-1*Ai-1*Ii-1)
*/
bool MGGPU_BuildAi(
	const uint3 resA,
	const MGGPU_Kernel3D<5> * Aprev,
	const MGGPU_InterpKernel * I,
	MGGPU_Kernel3D<5> * Anext,
	bool onDevice
);


void MGGPU_Residual_TopLevel(
	const uint3 res,
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_Volume & x,
	const MGGPU_Volume & f,
	MGGPU_Volume & r
);



#ifdef ___OLD


inline __device__ __host__ void MGGPU_A0(MGGPU_RestrictKernel & R, MGGPU_SystemTopKernel & A, MGGPU_InterpKernel & I) {

	//R*A


}

inline __device__ __host__ void MGGPU_CombineKernels(
	MGGPU_KernelPtr * a, int an,
	MGGPU_KernelPtr * b, int bn,
	int stride,
	MGGPU_KernelPtr * c
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
	const MGGPU_Volume & domainHalf,
	MGGPU_Kernel3D<4> * output
);

void MGGPU_GenerateAI0(
	const uint3 & Nres,
	const uint3 & Nhalfres,	
	const MGGPU_SystemTopKernel * A0,
	const MGGPU_Kernel3D<4> * IT0,
	MGGPU_Kernel3D<3> * output
);




bool MGGPU_CombineKernelsGeneric(
	const uint3 resArow,
	const uint3 resAcol,
	const uint3 resBrow,
	const uint3 resBcol,
	const MGGPU_KernelPtr A,
	const int Adim,
	const MGGPU_KernelPtr B,
	const int Bdim,
	MGGPU_KernelPtr C,
	const int Cdim,
	bool onDevice = true
);


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
	bool onDevice = true
);

bool MGGPU_CombineKernelsRestrict(
	const uint3 resArow,
	const uint3 resAcol,
	const uint3 resBrow,
	const uint3 resBcol,	
	const MGGPU_KernelPtr B,
	const int Bdim,
	MGGPU_KernelPtr C,
	bool onDevice = true
);

#endif