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
	return __hiloint2double(val.x, val.y);
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
		float vf0 = v0;
		float vf1 = v1;
		convertMaskKernel<float> <<< numBlocks, block >> > (res, surfIn, surfOut, vf0, vf1);
	}
	else if (type == TYPE_DOUBLE)
		convertMaskKernel<double> <<< numBlocks, block >> > (res, surfIn, surfOut, v0, v1);
	


}