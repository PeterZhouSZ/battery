#include "Volume.cuh"

#define VOLUME_VOX					\
uint3 vox = make_uint3(			\
		blockIdx.x * blockDim.x,	\
		blockIdx.y * blockDim.y,	\
		blockIdx.z * blockDim.z		\
	) + threadIdx;					\


#define VOLUME_VOX_GUARD(res)					\
	VOLUME_VOX									\
	if (vox.x >= res.x || vox.y >= res.y || vox.z >= res.z)	\
	return;			



__global__ void kernelErode(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut) {
	
	VOLUME_VOX_GUARD(res);

	float vals[6];	
	surf3Dread(&vals[0], surfIn, ((vox.x - 1 + res.x) % res.x) * sizeof(float), vox.y, vox.z);
	surf3Dread(&vals[1], surfIn, ((vox.x + 1) % res.x) * sizeof(float), vox.y, vox.z);
	surf3Dread(&vals[2], surfIn, vox.x * sizeof(float), (vox.y - 1 + res.y) % res.y , vox.z);
	surf3Dread(&vals[3], surfIn, vox.x * sizeof(float), (vox.y + 1) % res.y, vox.z);
	surf3Dread(&vals[4], surfIn, vox.x * sizeof(float), vox.y, (vox.z - 1 + res.z) % res.z);
	surf3Dread(&vals[5], surfIn, vox.x * sizeof(float), vox.y, (vox.z + 1) % res.z);

	float valCenter;
	surf3Dread(&valCenter, surfIn, vox.x * sizeof(float), vox.y, vox.z);

	float newVal = valCenter;
	bool isInside = ((vals[0] > 0.0f) && (vals[1] > 0.0f) && (vals[2] > 0.0f) && 
					(vals[3] > 0.0f) && (vals[4] > 0.0f) && (vals[5] > 0.0f));
	
	if (!isInside) {
		newVal = 0.0f;
	}	

	surf3Dwrite(newVal, surfOut, vox.x * sizeof(float), vox.y, vox.z);
}

void launchErodeKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut){

	uint3 block = make_uint3(8, 8, 8);
	uint3 numBlocks = make_uint3(
		(res.x / block.x) + 1,
		(res.y / block.y) + 1,
		(res.z / block.z) + 1
	);

	kernelErode << <numBlocks, block >> > (res, surfIn, surfOut);
}



template <int blockSize, int apron>
__global__ void kernelHeat(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut) {

	static_assert(apron * 2 < blockSize, "Apron must be less than blockSize / 2");
	const int N = blockSize;
	const int3 tid = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);

	//Sliding window of blockdim - 2*apron size
	const int3 vox = make_int3(
		blockIdx.x * (blockDim.x - 2*apron), 
		blockIdx.y * (blockDim.y - 2*apron), 
		blockIdx.z * (blockDim.z - 2*apron)
	) + tid - make_int3(apron);

	//Toroidal boundaries
	const int3 voxToroid = make_int3((vox.x + res.x) % res.x, (vox.y + res.y) % res.y, (vox.z + res.z) % res.z);
	
	//Read whole block into shared memory
	__shared__ float ndx[N][N][N];
	surf3Dread(
		&ndx[tid.x][tid.y][tid.z],
		surfIn,
		voxToroid.x * sizeof(float), voxToroid.y, voxToroid.z
	);
	__syncthreads();
	
	//Skip apron voxels
	if (tid.x < apron || tid.x >= N - apron ||
		tid.y < apron || tid.y >= N - apron ||
		tid.z < apron || tid.z >= N - apron ) return;

	//Skip outside voxels
	if (vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ||
		vox.x < 0 || vox.y < 0 || vox.z < 0)	return;

	/////////// Compute
		
	float oldVal = ndx[tid.x][tid.y][tid.z];
	/*
		Heaters
	*/
	int rad = 5;
	if (vox.x - res.x / 4 > res.x / 2 - rad && vox.y > res.y / 2 - rad && vox.z > res.z / 2 - rad &&
		vox.x - res.x / 4 < res.x / 2 + rad && vox.y < res.y / 2 + rad && vox.z < res.z / 2 + rad
		)
		oldVal += 1.5f;

	if (vox.x + res.x / 4 > res.x / 2 - rad && vox.y > res.y / 2 - rad && vox.z > res.z / 2 - rad &&
		vox.x + res.x / 4 < res.x / 2 + rad && vox.y < res.y / 2 + rad && vox.z < res.z / 2 + rad
		)
		oldVal += 1.5f;

	float dt = 0.1f;

	// New heat
	float newVal = oldVal + dt * (
		ndx[tid.x - 1][tid.y][tid.z] +
		ndx[tid.x + 1][tid.y][tid.z] +
		ndx[tid.x][tid.y - 1][tid.z] +
		ndx[tid.x][tid.y + 1][tid.z] +
		ndx[tid.x][tid.y][tid.z - 1] +
		ndx[tid.x][tid.y][tid.z + 1] -
		oldVal * 6.0f
		);

	surf3Dwrite(newVal, surfOut, vox.x * sizeof(float), vox.y, vox.z);	
	
}


void launchHeatKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut) {

	const int blockSize = 8;
	const int apron = 1;


	uint3 block = make_uint3(blockSize);	
	uint3 numBlocks = make_uint3(
		(res.x / (block.x - 2 * apron)) + 1,
		(res.y / (block.y - 2 * apron)) + 1,
		(res.z / (block.z - 2 * apron)) + 1
	);

	kernelHeat<blockSize,apron><< <numBlocks, block >> > (res, surfIn, surfOut);
}







__global__ void kernelBinarizeFloat(uint3 res, cudaSurfaceObject_t surfInOut, float threshold) {

	VOLUME_VOX_GUARD(res);

	float val = 0.0f;
	surf3Dread(&val, surfInOut, vox.x * sizeof(float), vox.y, vox.z);	

	val = (val < threshold) ? 0.0f : 1.0f;	

	surf3Dwrite(val, surfInOut, vox.x * sizeof(float), vox.y, vox.z);
}

template <typename T>
__global__ void kernelBinarizeUnsigned(uint3 res, cudaSurfaceObject_t surfInOut, T threshold) {

	VOLUME_VOX_GUARD(res);

	T val = 0;
	surf3Dread(&val, surfInOut, vox.x * sizeof(T), vox.y, vox.z);

	val = (val < threshold) ? T(0) : T(-1);

	surf3Dwrite(val, surfInOut, vox.x * sizeof(T), vox.y, vox.z);
}



void launchBinarizeKernel(uint3 res, cudaSurfaceObject_t surfInOut, PrimitiveType type, float threshold) {

	uint3 block = make_uint3(8, 8, 8);
	uint3 numBlocks = make_uint3(
		(res.x / block.x) + 1,
		(res.y / block.y) + 1,
		(res.z / block.z) + 1
	);

	if (type == TYPE_FLOAT)
		kernelBinarizeFloat << <numBlocks, block >> > (res, surfInOut, threshold);
	else if (type == TYPE_UCHAR)
		kernelBinarizeUnsigned<uchar> << <numBlocks, block >> > (res, surfInOut, uchar(threshold * 255));
	else
		exit(-1);
}


template <int blockSize, int apron>
__global__ void kernelDiffuse(DiffuseParams params) {	
	static_assert(apron * 2 < blockSize, "Apron must be less than blockSize / 2");
	const uint3 res = params.res;
	const int N = blockSize;
	const int3 tid = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);

	//Sliding window of blockdim - 2*apron size
	const int3 vox = make_int3(
		blockIdx.x * (blockDim.x - 2 * apron),
		blockIdx.y * (blockDim.y - 2 * apron),
		blockIdx.z * (blockDim.z - 2 * apron)
	) + tid - make_int3(apron);

	//Toroidal boundaries	

	//Read whole block into shared memory
	__shared__ float ndx[N][N][N];

	
	//Priority x > y > z (instead of 27 boundary values, just use 6)	
	Dir d = DIR_NONE;
	if (vox.x < 0) 		
		d = X_NEG;	
	else if (vox.x >= res.x) 
		d = X_POS;
	else if (vox.y < 0)
		d = Y_NEG;
	else if (vox.y >= res.y)
		d = Y_POS;
	else if (vox.z < 0)
		d = Z_NEG;
	else if (vox.z >= res.z)
		d = Z_POS;
	

	if (d != DIR_NONE) {
		ndx[tid.x][tid.y][tid.z] = params.boundaryValues[d];
	}
	else {
		surf3Dread(
			&ndx[tid.x][tid.y][tid.z],
			params.concetrationIn,
			vox.x * sizeof(float), vox.y, vox.z
		);
	}	
	__syncthreads();

	//If zero grad boundary cond, copy value from neighbour (after sync!)
	if (ndx[tid.x][tid.y][tid.z] == BOUNDARY_ZERO_GRADIENT) {		
		int3 neighVec = -dirVec(d);
		ndx[tid.x][tid.y][tid.z] = ndx[tid.x + neighVec.x][tid.y + neighVec.y][tid.z + neighVec.z];
	}

	__syncthreads();


	//Skip apron voxels
	if (tid.x < apron || tid.x >= N - apron ||
		tid.y < apron || tid.y >= N - apron ||
		tid.z < apron || tid.z >= N - apron) return;

	//Skip outside voxels
	if (vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ||
		vox.x < 0 || vox.y < 0 || vox.z < 0)	return;

	//Load battery value
	
	uchar mask = 0;
	surf3Dread(&mask, params.mask, vox.x * sizeof(uchar), vox.y, vox.z);
	if (mask > 0)
		return;
	
	/////////// Compute

	float oldVal = ndx[tid.x][tid.y][tid.z];
	/*
	Heaters
	*/
	int rad = 5;
	if (vox.x - res.x / 4 > res.x / 2 - rad && vox.y > res.y / 2 - rad && vox.z > res.z / 2 - rad &&
		vox.x - res.x / 4 < res.x / 2 + rad && vox.y < res.y / 2 + rad && vox.z < res.z / 2 + rad
		)
		oldVal += 1.5f;

	if (vox.x + res.x / 4 > res.x / 2 - rad && vox.y > res.y / 2 - rad && vox.z > res.z / 2 - rad &&
		vox.x + res.x / 4 < res.x / 2 + rad && vox.y < res.y / 2 + rad && vox.z < res.z / 2 + rad
		)
		oldVal += 1.5f;

	float dt = 0.1f;

	// New heat
	float newVal = oldVal + dt * (
		ndx[tid.x - 1][tid.y][tid.z] +
		ndx[tid.x + 1][tid.y][tid.z] +
		ndx[tid.x][tid.y - 1][tid.z] +
		ndx[tid.x][tid.y + 1][tid.z] +
		ndx[tid.x][tid.y][tid.z - 1] +
		ndx[tid.x][tid.y][tid.z + 1] -
		oldVal * 6.0f
		);

	surf3Dwrite(newVal, params.concetrationOut, vox.x * sizeof(float), vox.y, vox.z);

}

void launchDiffuseKernel(DiffuseParams params) {

	const int blockSize = 8;
	const int apron = 1;

	uint3 res = params.res;

	uint3 block = make_uint3(blockSize);
	uint3 numBlocks = make_uint3(
		(res.x / (block.x - 2 * apron)) + 1,
		(res.y / (block.y - 2 * apron)) + 1,
		(res.z / (block.z - 2 * apron)) + 1
	);

	kernelDiffuse<blockSize, apron> << <numBlocks, block >> > (params);
}