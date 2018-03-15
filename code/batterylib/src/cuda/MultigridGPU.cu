#include "MultigridGPU.cuh"
#include "Volume.cuh"



__device__ uint _getDirIndex(Dir dir) {
	switch (dir) {
	case X_POS:
	case X_NEG:
		return 0;
	case Y_POS:
	case Y_NEG:
		return 1;
	case Z_POS:
	case Z_NEG:
		return 2;
	}
	return uint(-1);
}


__device__ int _getDirSgn(Dir dir) {
	return -((dir % 2) * 2 - 1);
}

__device__ Dir _getDir(int index, int sgn) {
	sgn = (sgn + 1) / 2; // 0 neg, 1 pos
	sgn = 1 - sgn; // 1 neg, 0 pos
	return Dir(index * 2 + sgn);
}

template <typename T, typename VecType>
__device__ T & _at(VecType & vec, int index) {
	return ((T*)&vec)[index];
}
template <typename T, typename VecType>
__device__ const T & _at(const VecType & vec, int index){
	return ((T*)&vec)[index];
}


__device__ uint3 clampedVox(const uint3 & res, uint3 vox, Dir dir) {
	const int k = _getDirIndex(dir);
	int sgn = _getDirSgn(dir); //todo better

	const int newVox = _at<int>(vox, k) + sgn;
	const int & resK = _at<int>(res, k);
	if (newVox >= 0 && newVox < resK) {
		_at<int>(vox, k) = uint(newVox);		
	}
	return vox;
}

/*
	Templated surface write
*/
template <typename T>
__device__ void write(cudaSurfaceObject_t surf, const uint3 & vox, const T & val);


template<>
__device__ void write(cudaSurfaceObject_t surf, const uint3 & vox, const float & val) {
	surf3Dwrite(val, surf, vox.x * sizeof(float), vox.y, vox.z);
}

template<>
__device__ void write(cudaSurfaceObject_t surf, const uint3 & vox, const double & val) {
	const int2 * valInt = (int2*)&val;
	surf3Dwrite(*valInt, surf, vox.x * sizeof(int2), vox.y, vox.z);
}


/*
	Templated surface read (direct)
*/
template <typename T>
__device__ T read(cudaSurfaceObject_t surf, const uint3 & vox);

template<>
__device__ float read (cudaSurfaceObject_t surf, const uint3 & vox){
	float val;
	surf3Dread(&val, surf, vox.x * sizeof(float), vox.y, vox.z);
	return val;
}

template<>
__device__ double read(cudaSurfaceObject_t surf, const uint3 & vox) {
	int2 val;	
	surf3Dread(&val, surf, vox.x * sizeof(int2), vox.y, vox.z);
	return __hiloint2double(val.y, val.x);
}






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


	uint3 block = make_uint3(8, 8, 8);
	uint3 numBlocks = make_uint3(
		(res.x / block.x) + 1,
		(res.y / block.y) + 1,
		(res.z / block.z) + 1
	);

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

	uint3 block = make_uint3(2);
	uint3 numBlocks = make_uint3(
		(resDest.x / block.x) + 1,
		(resDest.y / block.y) + 1,
		(resDest.z / block.z) + 1
	);

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
	uint3 block = make_uint3(2);
	uint3 numBlocks = make_uint3(
		(resDest.x / block.x) + 1,
		(resDest.y / block.y) + 1,
		(resDest.z / block.z) + 1
	);

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
	//TODO
}

void launchWeightedInterpolationKernel(
	PrimitiveType type,
	cudaSurfaceObject_t surfSrc,
	cudaSurfaceObject_t surfWeight,
	uint3 resSrc,
	cudaSurfaceObject_t surfDest,
	uint3 resDest
) {

	uint3 block = make_uint3(2);
	uint3 numBlocks = make_uint3(
		(resDest.x / block.x) + 1,
		(resDest.y / block.y) + 1,
		(resDest.z / block.z) + 1
	);

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
	uint3 block = make_uint3(8);
	uint3 numBlocks = make_uint3(
		(params.res.x / block.x) + 1,
		(params.res.y / block.y) + 1,
		(params.res.z / block.z) + 1
	);

	if (params.type == TYPE_FLOAT){
		prepareSystemKernel<float> << < numBlocks, block >> >(params);
	}
	else if(params.type == TYPE_DOUBLE)
		prepareSystemKernel<double> << < numBlocks, block >> >(params);

}