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
};



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

using MGGPU_InterpKernel = MGGPU_Kernel3D<3>;
using MGGPU_RestrictKernel = MGGPU_Kernel3D<4>;
using MGGPU_DomainRestrictKernel = MGGPU_Kernel3D<2>;
using MGGPU_KernelPtr = double *;




inline __device__  MGGPU_InterpKernel MGGPU_GetInterpolationKernel(
	const MGGPU_Volume & domain,	
	const uint3 & vox, //vox in destination
	const uint3 & destRes, //should be double (if power of 2) of domain.res
	int dirIndex
) {

	/*
		Two spaces:
			source : n/2 (domain, domain.res)
			dest: n (vox, destRes)
	*/

	MGGPU_InterpKernel kernel;

	const int3 r = make_int3(vox.x % 2, vox.y % 2, vox.z % 2) * 2 - 1;
	const int3 voxi = make_int3(vox);
	const int3 voxSrc = make_int3(vox.x / 2, vox.y / 2, vox.z / 2);
	

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

	double w[8];
	double W = 0.0;
	for (int i = 0; i < 8; i++) {
		if (P[i] == 0) continue;
		w[i] = P[i];

		
		int3 voxSrcNew = voxSrc + offsets[i];
		if (_isValidPos(domain.res, voxSrcNew)) {
			w[i] *= read<double>(domain.surf, make_uint3(voxSrcNew)); //redundant conversion to uint, TODO better
		}	
		//Source voxel is outside of domain
		//P[i] > 0 then implies it's on dirichlet boundary
		//Therefore a nearest value has to be used
		else {
			
			//Change offset to nearest valid voxel from source
			int3 offset = offsets[i];			
			_at<int, int3>(offset, dirIndex) -= 1;			
			if (!_isValidPos(domain.res, voxSrcNew + r * offset)) {
				_at<int, int3>(offset, (dirIndex + 1) % 3) -= 1;				
			}

			if (!_isValidPos(domain.res, voxSrcNew + r * offset)) {
				_at<int, int3>(offset, (dirIndex + 2) % 3) -= 1;				
			}

			//Update src vox with new offset
			voxSrcNew = voxSrc + offset;

			//Read weight from source domain
			w[i] *= read<double>(domain.surf, make_uint3(voxSrcNew));			
		}

		W += w[i];
	}

	//Normalize weights
	for (auto i = 0; i < 8; i++) {
		w[i] /= W;
	}


	//Create 3^3 kernel
	memset(kernel.v, 0, 3 * 3 * 3 * sizeof(double));
	for (auto i = 0; i < 8; i++) {
		int3 kpos = make_int3(1, 1, 1) + offsets[i];
		kernel.v[kpos.x][kpos.y][kpos.z] = w[i];		
	}
	


	return kernel;
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
	
	memcpy(&kernel, w, 4 * 4 * 4 * sizeof(double));
	return kernel;
}


inline __device__ __host__ void MGGPU_CombineKernels(
	MGGPU_KernelPtr * a, int an,
	MGGPU_KernelPtr * b, int bn,
	MGGPU_KernelPtr * c
) {
	int cn = (an + bn - 1);		
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
	Convolves using a single kernel
*/
void MGGPU_Convolve(
	const MGGPU_Volume & in,
	MGGPU_KernelPtr kernel, int kn,
	const MGGPU_Volume & out
);


void MGGPU_GenerateSystemTopKernel(
	const MGGPU_Volume & domain,
	MGGPU_SystemTopKernel * A0,
	MGGPU_Volume & f
);