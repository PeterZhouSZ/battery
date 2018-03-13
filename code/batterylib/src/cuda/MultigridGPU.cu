#include "MultigridGPU.cuh"
#include "Volume.cuh"


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
	T val = binVal * v0 + (1 - binVal) * v1;
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