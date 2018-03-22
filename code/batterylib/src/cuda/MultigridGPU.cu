#include "MultigridGPU.cuh"
#include "Volume.cuh"


#include <stdio.h>




template <typename T>
__global__ void convertMaskKernel(
	uint3 res,
	cudaSurfaceObject_t surfIn,
	cudaSurfaceObject_t surfOut,
	T v0,
	T v1
) {

	VOLUME_VOX_GUARD(res);
	uchar maskVal;
	surf3Dread(&maskVal, surfIn, vox.x * sizeof(uchar), vox.y, vox.z);

	uchar binVal = maskVal / 255;
	T val = binVal * v1 + (1 - binVal) * v0;
	write<T>(surfOut, vox, val);

}

void launchConvertMaskKernel(
	PrimitiveType type, 
	uint3 res, 
	cudaSurfaceObject_t surfIn, 
	cudaSurfaceObject_t surfOut,
	double v0,
	double v1
) {


	BLOCKS3D(8, res);

	if (type == TYPE_FLOAT) {
		float vf0 = float(v0);
		float vf1 = float(v1);
		convertMaskKernel<float> <<< numBlocks, block >> > (res, surfIn, surfOut, vf0, vf1);
	}
	else if (type == TYPE_DOUBLE)
		convertMaskKernel<double> <<< numBlocks, block >> > (res, surfIn, surfOut, v0, v1);
	


}

__device__ size_t linearIndex(const uint3 & dim, const uint3 & pos) {
	return pos.x + dim.x * pos.y + dim.x * dim.y * pos.z;
}

template <typename T>
__global__ void restrictionKernel(cudaSurfaceObject_t surfSrc,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest, 
	T multiplier
) {
	VOLUME_VOX_GUARD(resDest);

	
	const uint3 voxSrc = vox * 2;	
	uint3 s = { 1, 1, 1 };
	if (voxSrc.x == resSrc.x - 1) s.x = 0;
	if (voxSrc.y == resSrc.y - 1) s.y = 0;
	if (voxSrc.z == resSrc.z - 1) s.z = 0;
	
	T val = read<T>(surfSrc, voxSrc) +
		read<T>(surfSrc, voxSrc + s * make_uint3(1, 0, 0)) +
		read<T>(surfSrc, voxSrc + s * make_uint3(0, 1, 0)) +
		read<T>(surfSrc, voxSrc + s * make_uint3(1, 1, 0)) +
		read<T>(surfSrc, voxSrc + s * make_uint3(0, 0, 1)) +
		read<T>(surfSrc, voxSrc + s * make_uint3(1, 0, 1)) +
		read<T>(surfSrc, voxSrc + s * make_uint3(0, 1, 1)) +
		read<T>(surfSrc, voxSrc + s * make_uint3(1, 1, 1));

	val *= T(1.0 / 8.0) * multiplier;

	write<T>(surfDest, vox, val);
}

void launchRestrictionKernel(
	PrimitiveType type,
	cudaSurfaceObject_t surfSrc,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest,
	double multiplier
) {

	BLOCKS3D(2, resDest);
	if (type == TYPE_FLOAT)
		restrictionKernel<float> << < numBlocks, block >> > (surfSrc,resSrc,surfDest,resDest, float(multiplier));
	else if (type == TYPE_DOUBLE)
		restrictionKernel<double> << < numBlocks, block >> > (surfSrc, resSrc, surfDest, resDest, multiplier);

}


template <typename T>
__global__ void weightedRestrictionKernel(
	cudaSurfaceObject_t surfSrc,
	cudaSurfaceObject_t surfWeight,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest
) {
	VOLUME_VOX_GUARD(resDest);


	const uint3 voxSrc = vox * 2;
	uint3 s = { 1, 1, 1 };
	if (voxSrc.x == resSrc.x - 1) s.x = 0;
	if (voxSrc.y == resSrc.y - 1) s.y = 0;
	if (voxSrc.z == resSrc.z - 1) s.z = 0;

	const uint3 voxSrcOff[8] = {
		voxSrc + 0,
		voxSrc + s * make_uint3(1, 0, 0),
		voxSrc + s * make_uint3(0, 1, 0),
		voxSrc + s * make_uint3(1, 1, 0),
		voxSrc + s * make_uint3(0, 0, 1),
		voxSrc + s * make_uint3(1, 0, 1),
		voxSrc + s * make_uint3(0, 1, 1),
		voxSrc + s * make_uint3(1, 1, 1)
	};

	const T weights[8] = {
		read<T>(surfWeight, voxSrcOff[0]),
		read<T>(surfWeight, voxSrcOff[1]),
		read<T>(surfWeight, voxSrcOff[2]),
		read<T>(surfWeight, voxSrcOff[3]),
		read<T>(surfWeight, voxSrcOff[4]),
		read<T>(surfWeight, voxSrcOff[5]),
		read<T>(surfWeight, voxSrcOff[6]),
		read<T>(surfWeight, voxSrcOff[7])
	};
	
	const T W = weights[0] + weights[1] + weights[2] + weights[3] + weights[4] + weights[5] + weights[6] + weights[7];
	
	//todo weights -> constant memory? vs tex?
	T val = read<T>(surfSrc, voxSrcOff[0]) * weights[0] +
		read<T>(surfSrc, voxSrcOff[1]) *  weights[1] +
		read<T>(surfSrc, voxSrcOff[2]) *  weights[2] +
		read<T>(surfSrc, voxSrcOff[3]) *  weights[3] +
		read<T>(surfSrc, voxSrcOff[4]) *  weights[4] +
		read<T>(surfSrc, voxSrcOff[5]) *  weights[5] +
		read<T>(surfSrc, voxSrcOff[6]) *  weights[6] +
		read<T>(surfSrc, voxSrcOff[7]) *  weights[7];
	
	val /= W;

	write<T>(surfDest, vox, val);
}


void launchWeightedRestrictionKernel(
	PrimitiveType type,
	cudaSurfaceObject_t surfSrc,
	cudaSurfaceObject_t surfWeight,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest
) {
	BLOCKS3D(2, resDest);

	if (type == TYPE_FLOAT)
		weightedRestrictionKernel<float> << < numBlocks, block >> > (surfSrc,surfWeight, resSrc, surfDest, resDest);
	else if (type == TYPE_DOUBLE)
		weightedRestrictionKernel<double> << < numBlocks, block >> > (surfSrc, surfWeight, resSrc, surfDest, resDest);
}

template <typename T>
__global__ void weightedInterpolationKernel(
	cudaSurfaceObject_t surfSrc,
	cudaSurfaceObject_t surfWeight,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest
) {
	VOLUME_VOX_GUARD(resDest);
	
	const int3 voxSrc = { int(vox.x) / 2, int(vox.y) / 2, int(vox.z) / 2 };
	
	//Direction
	// -1 for even, 1 for odd 
	int3 s = make_int3(vox.x % 2, vox.y % 2, vox.z % 2) * 2 - 1;
	if (vox.x == resDest.x - 1 || vox.x == 0) s.x = 0;
	if (vox.y == resDest.y - 1 || vox.y == 0) s.y = 0;
	if (vox.z == resDest.z - 1 || vox.z == 0) s.z = 0;

	const uint3 voxSrcOff[8] = {
		make_uint3(voxSrc + 0),
		make_uint3(voxSrc + s * make_int3(1, 0, 0)),
		make_uint3(voxSrc + s * make_int3(0, 1, 0)),
		make_uint3(voxSrc + s * make_int3(1, 1, 0)),
		make_uint3(voxSrc + s * make_int3(0, 0, 1)),
		make_uint3(voxSrc + s * make_int3(1, 0, 1)),
		make_uint3(voxSrc + s * make_int3(0, 1, 1)),
		make_uint3(voxSrc + s * make_int3(1, 1, 1))
	};


	const T w[8] = {
		read<T>(surfWeight, voxSrcOff[0]),
		read<T>(surfWeight, voxSrcOff[1]),
		read<T>(surfWeight, voxSrcOff[2]),
		read<T>(surfWeight, voxSrcOff[3]),
		read<T>(surfWeight, voxSrcOff[4]),
		read<T>(surfWeight, voxSrcOff[5]),
		read<T>(surfWeight, voxSrcOff[6]),
		read<T>(surfWeight, voxSrcOff[7])
	};

	const T vals[8] = {
		read<T>(surfSrc, voxSrcOff[0]),
		read<T>(surfSrc, voxSrcOff[1]),
		read<T>(surfSrc, voxSrcOff[2]),
		read<T>(surfSrc, voxSrcOff[3]),
		read<T>(surfSrc, voxSrcOff[4]),
		read<T>(surfSrc, voxSrcOff[5]),
		read<T>(surfSrc, voxSrcOff[6]),
		read<T>(surfSrc, voxSrcOff[7])
	};


	const T a = T(1.0 / 4.0);
	const T ainv = T(1.0) - a;

	T w_x0y0 = ainv * w[0] + a * w[4];
	T w_x0y1 = ainv * w[2] + a * w[6];
	T w_x0 = ainv * w_x0y0 + a * w_x0y1;

	T w_x1y0 = ainv * w[1] + a * w[5];
	T w_x1y1 = ainv * w[3] + a * w[7];
	T w_x1 = ainv * w_x1y0 + a * w_x1y1;

	T w_val = ainv * w_x0 + a * w_x1;


	T x0y0 = ainv * vals[0] * w[0] + a * vals[4] * w[4];
	T x0y1 = ainv * vals[2] * w[2] + a * vals[6] * w[6];
	T x0 = ainv * x0y0 + a * x0y1;

	T x1y0 = ainv * vals[1] * w[1] + a * vals[5] * w[5];
	T x1y1 = ainv * vals[3] * w[3] + a * vals[7] * w[7];
	T x1 = ainv * x1y0 + a * x1y1;

	T val = ainv * x0 + a * x1;
	val /= w_val;
	
	write<T>(surfDest, vox, val);

}

void launchWeightedInterpolationKernel(
	PrimitiveType type,
	cudaSurfaceObject_t surfSrc,
	cudaSurfaceObject_t surfWeight,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest
) {

	BLOCKS3D(2, resDest);

	if (type == TYPE_FLOAT)
		weightedInterpolationKernel<float> << < numBlocks, block >> > (surfSrc, surfWeight, resSrc, surfDest, resDest);
	else if (type == TYPE_DOUBLE)
		weightedInterpolationKernel<double> << < numBlocks, block >> > (surfSrc, surfWeight, resSrc, surfDest, resDest);
}





template <typename T>
__global__ void prepareSystemKernel(LinSysParams params) {
	VOLUME_VOX_GUARD(params.res);
		
	const T highConc = T(1.0);
	const T lowConc = T(0.0);
	const T concetrationBegin = (_getDirSgn(params.dir) == 1) ? highConc : lowConc;
	const T concetrationEnd = (_getDirSgn(params.dir) == 1) ? lowConc : highConc;

	const T cellDim[3] = { params.cellDim.x,params.cellDim.y,params.cellDim.z };
	const T faceArea[3] = { cellDim[1] * cellDim[2],cellDim[0] * cellDim[2],cellDim[0] * cellDim[1] };

	const size_t i = linearIndex(params.res, vox);
	const size_t rowI = i * 7;
	T Di = read<T>(params.surfD, vox);
		
	T Dneg[3] = {
		(read<T>(params.surfD, clampedVox(params.res, vox, X_NEG)) + Di) * T(0.5),
		(read<T>(params.surfD, clampedVox(params.res, vox, Y_NEG)) + Di) * T(0.5),
		(read<T>(params.surfD, clampedVox(params.res, vox, Z_NEG)) + Di) * T(0.5)
	};

	T Dpos[3] = {
		(read<T>(params.surfD, clampedVox(params.res, vox, X_POS)) + Di) * T(0.5),
		(read<T>(params.surfD, clampedVox(params.res, vox, Y_POS)) + Di) * T(0.5),
		(read<T>(params.surfD, clampedVox(params.res, vox, Z_POS)) + Di) * T(0.5)
	};

	

	T coeffs[7];
	bool useInMatrix[7];

	coeffs[DIR_NONE] = T(0);
	useInMatrix[DIR_NONE] = true;

	for (uint j = 0; j < DIR_NONE; j++) {
		const uint k = _getDirIndex(Dir(j));
		const int sgn = _getDirSgn(Dir(j));
		const T Dface = (sgn == -1) ? Dneg[k] : Dpos[k];

		T cellDist[3] = { cellDim[0],cellDim[1],cellDim[2] };
		useInMatrix[j] = true;
		
		if ((_at<uint>(vox,k) == 0 && sgn == -1) || 
			(_at<uint>(vox, k) == _at<uint>(params.res, k) - 1 && sgn == 1)
			) {
			cellDist[k] = cellDim[k] * T(0.5);
			useInMatrix[j] = false;
		}

		coeffs[j] = (Dface * faceArea[k]) / cellDist[k];

		//Subtract from diagonal
		if (useInMatrix[j] || k == params.dirPrimary)
			coeffs[DIR_NONE] -= coeffs[j]; 
	}

	/*
		Right hand side
	*/

	const uint primaryRes = ((uint*)&params.res)[params.dirPrimary];
	T rhs = T(0);
	if (_at<uint>(vox, params.dirPrimary) == 0) {
		Dir dir = _getDir(params.dirPrimary, -1);
		rhs -= coeffs[dir] * concetrationBegin;
	}
	else if (_at<uint>(vox, params.dirPrimary) == primaryRes - 1) {
		Dir dir = _getDir(params.dirPrimary, 1);
		rhs -= coeffs[dir] * concetrationEnd;
	}
	//Write to F
	write<T>(params.surfF, vox, rhs);


	//X initial guess
	T xGuess = T(0);	
	if (_getDirSgn(params.dir) == 1)
		xGuess = 1.0f - (_at<uint>(vox, params.dirPrimary) / T(primaryRes + 1));
	else
		xGuess = (_at<uint>(vox, params.dirPrimary) / T(primaryRes + 1));

	//Write to X
	write<T>(params.surfX, vox, xGuess);

	for (uint j = 0; j < DIR_NONE; j++) {
		if (!useInMatrix[j]) 
			coeffs[j] = T(0);
	}

	T * matrixRow = ((T*)params.matrixData) + rowI;
	matrixRow[0] = coeffs[Z_NEG];
	matrixRow[1] = coeffs[Y_NEG];
	matrixRow[2] = coeffs[X_NEG];
	matrixRow[3] = coeffs[DIR_NONE];
	matrixRow[4] = coeffs[X_POS];
	matrixRow[5] = coeffs[Y_POS];
	matrixRow[6] = coeffs[Z_POS];


}


void launchPrepareSystemKernel(LinSysParams params)
{
	BLOCKS3D(8, params.res);

	if (params.type == TYPE_FLOAT){
		prepareSystemKernel<float> << < numBlocks, block >> >(params);
	}
	else if(params.type == TYPE_DOUBLE)
		prepareSystemKernel<double> << < numBlocks, block >> >(params);

}

template <typename T>
__global__ void clearSurfaceKernel(cudaSurfaceObject_t surf, uint3 res, T val) {
	VOLUME_VOX_GUARD(res);
	write<T>(surf, vox, val);
}

void clearSurface(PrimitiveType type, cudaSurfaceObject_t surf, uint3 res, void * val) {	
	BLOCKS3D(8, res);	
	if (type == TYPE_FLOAT) {
		clearSurfaceKernel<float> << < numBlocks, block >> >(surf,res,*((float *)val));
	}
	else if (type == TYPE_DOUBLE)
		clearSurfaceKernel<double> << < numBlocks, block >> >(surf,res, *((double *)val));
}

template <typename T>
__global__ void surfaceAdditionKernel(cudaSurfaceObject_t A, cudaSurfaceObject_t B, uint3 res) {
	VOLUME_VOX_GUARD(res);
	T val = read<T>(A, vox) + read<T>(B, vox);
	write<T>(A, vox, val);
}




void surfaceAddition(PrimitiveType type, cudaSurfaceObject_t A, cudaSurfaceObject_t B, uint3 res) {
	BLOCKS3D(8, res);

	if (type == TYPE_FLOAT) 
		surfaceAdditionKernel<float> << < numBlocks, block >> >(A,B,res);
	
	else if (type == TYPE_DOUBLE)
		surfaceAdditionKernel<double> << < numBlocks, block >> >(A, B, res);
}

template <typename T>
__global__ void surfaceSubtractionKernel(cudaSurfaceObject_t A, cudaSurfaceObject_t B, uint3 res) {
	VOLUME_VOX_GUARD(res);
	T val = read<T>(A, vox) - read<T>(B, vox);
	write<T>(A, vox, val);
}

void surfaceSubtraction(PrimitiveType type, cudaSurfaceObject_t A, cudaSurfaceObject_t B, uint3 res) {
	BLOCKS3D(8, res);
	if (type == TYPE_FLOAT) {
		surfaceSubtractionKernel<float> << < numBlocks, block >> >(A, B, res);
	}
	else if (type == TYPE_DOUBLE)
		surfaceSubtractionKernel<double> << < numBlocks, block >> >(A, B, res);
}

template <typename T>
__global__ void residualKernel(uint3 res,
	cudaSurfaceObject_t surfR,
	cudaSurfaceObject_t surfF,
	cudaSurfaceObject_t surfX,
	T * matrixData) {
	VOLUME_VOX_GUARD(res);

	//r = f - A*x

	const size_t i = linearIndex(res, vox);
	const size_t rowI = i * 7;
	const T * row = matrixData + rowI;

	//const uint3 stride = { 1, res.x, res.x*res.y };
	const uint3 voxCol[7] = {
		vox - make_uint3(0,0,1),
		vox - make_uint3(0,1,0),
		vox - make_uint3(1,0,0),
		vox,
		vox + make_uint3(1,0,0),
		vox + make_uint3(0,1,0),
		vox + make_uint3(0,0,1)
	};

	//const uint3 vox
	T sumProduct = T(0.0);	
	for (auto k = 0; k < 7; k++) {
		T coeff = row[k];		
		if (coeff == T(0.0)) continue; //coeffs outside boundary are also 0

		sumProduct += coeff * read<T>(surfX, voxCol[k]);
	}

	T fval = read<T>(surfF, vox);	
	T residue = fval - sumProduct;	
	//residue = abs(residue) * 10000;
	//T residue = coeffSum;
	write<T>(surfR, vox, residue);	
}

//r = f - A*x
void residual(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t surfR,
	cudaSurfaceObject_t surfF,
	cudaSurfaceObject_t surfX,
	void * matrixData
) {	

	BLOCKS3D(8, res);
	if (type == TYPE_FLOAT)
		residualKernel<float> << < numBlocks, block >> >(res,surfR,surfF,surfX, (float*)matrixData);	
	else if (type == TYPE_DOUBLE)
		residualKernel<double> << < numBlocks, block >> >(res, surfR, surfF, surfX, (double*)matrixData);
}



///////////////////////////////////////////////////

template <typename T, int dir, int sgn, bool alternate>  
__global__ void gaussSeidelLineKernel(
	uint3 res,
	cudaSurfaceObject_t surfB,
	cudaSurfaceObject_t surfX,
	const T * matrixData
) {

	VOLUME_VOX;
	
	uint primDim = _at<uint>(res, dir);
	const int secDirs[2] = {
		(dir + 1) % 3,
		(dir + 2) % 3
	};

	_at<uint>(vox, secDirs[0]) *= 2;
	if (!alternate)
		_at<uint>(vox, secDirs[0]) += _at<uint>(vox, secDirs[1]) % 2;
	else
		_at<uint>(vox, secDirs[0]) += 1 - _at<uint>(vox, secDirs[1]) % 2;
	

	if (_at<uint>(vox, secDirs[0]) >= _at<uint>(res, secDirs[0]) ||
		_at<uint>(vox, secDirs[1]) >= _at<uint>(res, secDirs[1]))
		return;
	
	const int begin = (sgn == -1) ? int(primDim) - 1 : 0;
	const int end = (sgn == -1) ? -1 : int(primDim);

	

	for (int i = begin; i != end; i+=sgn) {
		uint3 voxi = {vox.x, vox.y, vox.z};
		_at<uint>(voxi, dir) = i;
		const size_t rowI = linearIndex(res, voxi) * 7;

		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
			if (blockIdx.x == 0 && blockIdx.y == 2 && blockIdx.z == 0) {
				//printf("%d + %d ... %d %d %d\n",i, sgn, voxi.x, voxi.y, vox.z);
			}

		}

		

		const T * row = matrixData + rowI;
		const T rowVals[7] = {
			row[0],row[1],row[2],row[3],row[4],row[5],row[6]
		};

		

		T sum = T(0);	
		
		sum += (rowVals[0] == T(0)) ? T(0) : rowVals[0] * read<T>(surfX, voxi - make_uint3(0, 0, 1));
		sum += (rowVals[1] == T(0)) ? T(0) : rowVals[1] * read<T>(surfX, voxi - make_uint3(0, 1, 0));
		sum += (rowVals[2] == T(0)) ? T(0) : rowVals[2] * read<T>(surfX, voxi - make_uint3(1, 0, 0));
		T diag = rowVals[3];
		sum += (rowVals[4] == T(0)) ? T(0) : rowVals[4] * read<T>(surfX, voxi + make_uint3(1, 0, 0));
		sum += (rowVals[5] == T(0)) ? T(0) : rowVals[5] * read<T>(surfX, voxi + make_uint3(0, 1, 0));
		sum += (rowVals[6] == T(0)) ? T(0) : rowVals[6] * read<T>(surfX, voxi + make_uint3(0, 0, 1));

		

		T bval = read<T>(surfB, voxi);
		T newVal = (bval - sum) / diag;

		//if (rowI == 0)
			//printf("i0 ( %d %d %d) ... %d %d %d ... %f, %f\n",dir,sgn,alternate, threadIdx.x, threadIdx.y, threadIdx.z, sum, newVal);

		write<T>(surfX, voxi, newVal);	
		//write<T>(surfX, voxi, diag);
	}


}

template <typename T>
void solveGaussSeidelForType(const SmootherParams & params) {

	T bsq = T(0.0);
	launchReduceKernel(
		params.type,
		REDUCE_OP_SQUARESUM,
		params.res,
		params.surfB,
		params.auxBufferGPU,
		params.auxBufferCPU,
		&bsq
	);

	T tol_error = T(1);

	for (auto i = 0; i < params.maxIter; ++i) {


		

		//k 0,2,5 dirs
		//gaussSeidelStepZebra<k>(A, b, x, dim, Dir(k));
		
		uint3 block;
		uint3 numBlocks;
		
		

		
		//X
		block = make_uint3(1, 8, 8);
		numBlocks = make_uint3(
			1,
			((params.res.y / 2 + (block.y - 1)) / block.y),
			((params.res.z + (block.z - 1)) / block.z)
		);
		gaussSeidelLineKernel<T, 0, 1, false> << <numBlocks, block >> > (
			params.res, params.surfB, params.surfX, ((T*)params.matrixData)
		);
		
		gaussSeidelLineKernel<T, 0, 1, true> << <numBlocks, block >> > (
			params.res, params.surfB, params.surfX, ((T*)params.matrixData)
			);
		
		//Y
		block = make_uint3(8, 1, 8);
		numBlocks = make_uint3(			
			((params.res.x + (block.x - 1)) / block.x),
			1,
			((params.res.z / 2 + (block.z - 1)) / block.z)
		);
		gaussSeidelLineKernel<T, 1, 1, false> << <numBlocks, block >> > (
			params.res, params.surfB, params.surfX, ((T*)params.matrixData)
			);
		
		gaussSeidelLineKernel<T, 1, 1, true> << <numBlocks, block >> > (
			params.res, params.surfB, params.surfX, ((T*)params.matrixData)
			);
		
		//Z
		block = make_uint3(8, 8, 1);
		numBlocks = make_uint3(			
			((params.res.x / 2 + (block.x - 1)) / block.y),
			((params.res.y + (block.y - 1)) / block.y),
			1
		);
		gaussSeidelLineKernel<T, 2, 1, false> << <numBlocks, block >> > (
			params.res, params.surfB, params.surfX, ((T*)params.matrixData)
			);
		
		gaussSeidelLineKernel<T, 2, 1, true> << <numBlocks, block >> > (
			params.res, params.surfB, params.surfX, ((T*)params.matrixData)
			);
		

		//printf("----------\n");

		//r = b - A*x
		residual(
			params.type,
			params.res,
			params.surfR,
			params.surfB,
			params.surfX,
			params.matrixData
		);

		T rsq = T(0.0);
		launchReduceKernel(
			params.type,
			REDUCE_OP_SQUARESUM,
			params.res,
			params.surfR,
			params.auxBufferGPU,
			params.auxBufferCPU,
			&rsq
		);

		tol_error = sqrt(rsq / bsq);
	

	//	printf("tol error, i %d: %f\n", i, tol_error);
		if (tol_error <= *((T*)params.tolerance))
			break;
	
	}

	*((T*)params.errorOut) = tol_error;
	
}

void solveGaussSeidel(
	const SmootherParams & params
) {
	if (params.type == TYPE_FLOAT)
		solveGaussSeidelForType<float>(
			params
		);
	else if(params.type == TYPE_DOUBLE)
		solveGaussSeidelForType<double>(
			params
			);

}




/////////////////////////////////

template <typename T>
__global__ void jacobiKernel(
	uint3 res,
	cudaSurfaceObject_t surfB,
	cudaSurfaceObject_t surfX,
	cudaSurfaceObject_t surfXnew,
	const T * matrixData
) {
	VOLUME_VOX_GUARD(res);

	const size_t rowI = linearIndex(res, vox) * 7;

	const T * row = matrixData + rowI;
	const T rowVals[7] = {
		row[0],row[1],row[2],row[3],row[4],row[5],row[6]
	};
	
	T sum = T(0);

	sum += (rowVals[0] == T(0)) ? T(0) : rowVals[0] * read<T>(surfX, vox - make_uint3(0, 0, 1));
	sum += (rowVals[1] == T(0)) ? T(0) : rowVals[1] * read<T>(surfX, vox - make_uint3(0, 1, 0));
	sum += (rowVals[2] == T(0)) ? T(0) : rowVals[2] * read<T>(surfX, vox - make_uint3(1, 0, 0));
	T diag = rowVals[3];
	sum += (rowVals[4] == T(0)) ? T(0) : rowVals[4] * read<T>(surfX, vox + make_uint3(1, 0, 0));
	sum += (rowVals[5] == T(0)) ? T(0) : rowVals[5] * read<T>(surfX, vox + make_uint3(0, 1, 0));
	sum += (rowVals[6] == T(0)) ? T(0) : rowVals[6] * read<T>(surfX, vox + make_uint3(0, 0, 1));

	T bval = read<T>(surfB, vox);
	T newVal = (bval - sum) / diag;

	write<T>(surfXnew, vox, newVal);

}


template <typename T>
void solveJacobiForType(const SmootherParams & params) {

	T bsq = T(0.0);
	launchReduceKernel(
		params.type,
		REDUCE_OP_SQUARESUM,
		params.res,
		params.surfB,
		params.auxBufferGPU,
		params.auxBufferCPU,
		&bsq
	);

	T tol_error = T(1);

	cudaSurfaceObject_t X[2] = {
		params.surfX,
		params.surfXtemp
	};

	

	for (auto i = 0; i < params.maxIter; ++i) {

		
		//kernel launch
		BLOCKS3D(8, params.res);

		for (auto k = 0; k < 2; k++) {
			jacobiKernel<T> << <numBlocks, block >> > (
				params.res,
				params.surfB,
				X[k % 2],
				X[(k + 1) % 2],
				(T*)params.matrixData
				);
		}

		//r = b - A*x
		residual(
			params.type,
			params.res,
			params.surfR,
			params.surfB,
			params.surfX,
			params.matrixData
		);

		T rsq = T(0.0);
		launchReduceKernel(
			params.type,
			REDUCE_OP_SQUARESUM,
			params.res,
			params.surfR,
			params.auxBufferGPU,
			params.auxBufferCPU,
			&rsq
		);

		tol_error = sqrt(rsq / bsq);

		if (tol_error <= *((T*)params.tolerance))
			break;

	}

	*((T*)params.errorOut) = tol_error;

}


void solveJacobi(
	const SmootherParams & params
) {
	if (params.type == TYPE_FLOAT)
		solveJacobiForType<float>(
			params
			);
	else if (params.type == TYPE_DOUBLE)
		solveJacobiForType<double>(
			params
			);

}