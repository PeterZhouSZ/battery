#pragma once
#include "Volume.cuh"


/*
	CUDA implementation of generating Ab = x linear system for steady state diffusion	
*/

struct STRUCT_ALIGN(16) CUDA_Stencil_7 {
	double v[7];
};

//Construction params
struct LinearSys_SysParams {
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


//Convolve by system top kernel
inline double  __device__  convolve3D_SystemTop(
	const int3 & ivox,
	const CUDA_Stencil_7 & k,
	const CUDA_Volume & x
) {

	//Assumes ivox is inside the boundary

	double sum = 0.0;

	if (ivox.z - 1 >= 0)
		sum += k.v[Z_NEG] * read<double>(x.surf, ivox + make_int3(0, 0, -1));

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


/*
Generates domain from mask and two double values
*/
void LinearSys_GenerateDomain(
	const CUDA_Volume & binaryMask,
	double value_zero,
	double value_one,
	CUDA_Volume & output
);

/*
Generates 7-point stencil of linear system as a function of the domain
*/
void LinearSys_GenerateSystem(
	const CUDA_Volume & domain,
	CUDA_Stencil_7 * A0, //A matrix, stored as an array of 7-pt stencils (z major order)
	CUDA_Volume & f, //rhs of lin sys,
	CUDA_Volume * x = nullptr //optional guess
);


/*
	Commits system parameters to CUDA constant memory
*/
bool LinearSys_commitSysParams(const LinearSys_SysParams & sysparams);


/*
	Inverts A diagonal to vector
*/
void LinearSys_InvertSystemDiagTo(
	const CUDA_Stencil_7 * A,
	CUDA_Volume & ainvert
);

/*
	Calculates residual r = Ax - b
	TODO: use shared memory optimized LinearSys_MatrixVectorProduct for residual computation
*/
void LinearSys_Residual(	
	const CUDA_Stencil_7 * A,
	const CUDA_Volume & x,
	const CUDA_Volume & f,
	CUDA_Volume & r
);

/*
	Calculates b = Ax
*/
void LinearSys_MatrixVectorProduct(
	const CUDA_Stencil_7 * A,
	const CUDA_Volume & x,
	CUDA_Volume & b
);