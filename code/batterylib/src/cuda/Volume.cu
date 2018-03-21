#include "Volume.cuh"
#include <stdio.h>


	



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
	__shared__ float Ddx[N][N][N];

	
	//Priority x > y > z (instead of 27 boundary values, just use 6)	
	Dir dir = DIR_NONE;
	if (vox.x < 0) 		
		dir = X_NEG;	
	else if (vox.x >= res.x) 
		dir = X_POS;
	else if (vox.y < 0)
		dir = Y_NEG;
	else if (vox.y >= res.y)
		dir = Y_POS;
	else if (vox.z < 0)
		dir = Z_NEG;
	else if (vox.z >= res.z)
		dir = Z_POS;
	

	if (dir != DIR_NONE) {
		ndx[tid.x][tid.y][tid.z] = params.boundaryValues[dir];
		Ddx[tid.x][tid.y][tid.z] = BOUNDARY_ZERO_GRADIENT;
	}
	else {
		surf3Dread(
			&ndx[tid.x][tid.y][tid.z],
			params.concetrationIn,
			vox.x * sizeof(float), vox.y, vox.z
		);

		uchar maskVal;
		surf3Dread(
			&maskVal,
			params.mask,
			vox.x * sizeof(uchar), vox.y, vox.z
		);
		if (maskVal == 0)
			Ddx[tid.x][tid.y][tid.z] = params.zeroDiff;
		else
			Ddx[tid.x][tid.y][tid.z] = params.oneDiff;


	}	
	__syncthreads();

	//If zero grad boundary cond, copy value from neighbour (after sync!)
	if (ndx[tid.x][tid.y][tid.z] == BOUNDARY_ZERO_GRADIENT) {		
		int3 neighVec = -dirVec(dir);
		ndx[tid.x][tid.y][tid.z] = ndx[tid.x + neighVec.x][tid.y + neighVec.y][tid.z + neighVec.z];
	}

	if (Ddx[tid.x][tid.y][tid.z] == BOUNDARY_ZERO_GRADIENT) {
		int3 neighVec = -dirVec(dir);
		Ddx[tid.x][tid.y][tid.z] = Ddx[tid.x + neighVec.x][tid.y + neighVec.y][tid.z + neighVec.z];
	}
	//TODO: test what is faster -> double read from global memory, or copy within shared with extra threadsync

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

	//Diffusion coeff
	float D = (mask == 0) ? params.zeroDiff : params.oneDiff;	



	
	///
	{
		float dx = params.voxelSize;

		const float D = Ddx[tid.x][tid.y][tid.z];
		const float3 D3 = make_float3(D);

		const float3 Dneg = lerp(
			D3,
			make_float3(Ddx[tid.x - 1][tid.y][tid.z], Ddx[tid.x][tid.y - 1][tid.z], Ddx[tid.x][tid.y][tid.z-1]),
			(dx * 0.5f)
		);

		const float3 Dpos = lerp(
			D3,
			make_float3(Ddx[tid.x + 1][tid.y][tid.z], Ddx[tid.x][tid.y + 1][tid.z], Ddx[tid.x][tid.y][tid.z + 1]),			
			(dx * 0.5f)
		);	


		const float3 C = make_float3(ndx[tid.x][tid.y][tid.z]);

		const float3 Cneg = lerp(
			C,
			make_float3(ndx[tid.x - 1][tid.y][tid.z], ndx[tid.x][tid.y - 1][tid.z],	ndx[tid.x][tid.y][tid.z - 1]),
			dx
		);

		const float3 Cpos = lerp(
			C,
			make_float3(ndx[tid.x + 1][tid.y][tid.z], ndx[tid.x][tid.y + 1][tid.z], ndx[tid.x][tid.y][tid.z + 1]),
			dx
		);
		
		float dt = dx*dx * (1.0f / (2.0f * min(params.zeroDiff, params.oneDiff)));


		//https://math.stackexchange.com/questions/1949795/explicit-finite-difference-scheme-for-nonlinear-diffusion
		//float3 dc = (dt / (dx*dx)) * (Dpos * (Cpos - C) - Dneg * (C - Cneg));

		float3 dc = Dneg * Cneg + Dpos * Cpos - C * (Dneg + Dpos);

		//float3 dc = D * (Cpos - 2* C + Cneg) + (Dneg - Dpos) * ()

		//if (vox.x == 2 && vox.y == 10 && vox.z == 10) {
			//printf("dt: %f\n", dt);
		//}
			//printf("c: %f, D: %.9f dc: %f %f %f, Dneg: %f %f %f\n",C.x, D, dc.x, dc.y, dc.z, Dneg.x, Dneg.y, Dneg.z);

		float DX = 1.0f / res.x;

		float newVal = C.x + (dc.x + dc.y + dc.z);
		

		surf3Dwrite(newVal, params.concetrationOut, vox.x * sizeof(float), vox.y, vox.z);


		return;

		//float3 dD2 = make_float3(
		//	Ddx[tid.x - 1][tid.y][tid.z] + 2.0f * oldVal + Ddx[tid.x + 1][tid.y][tid.z],
		//	Ddx[tid.x][tid.y - 1][tid.z] + 2.0f * oldVal + Ddx[tid.x][tid.y + 1][tid.z],
		//	Ddx[tid.x][tid.y][tid.z - 1] + 2.0f * oldVal + Ddx[tid.x][tid.y][tid.z + 1]
		//);

		//float3 dc2 = make_float3(
		//	ndx[tid.x - 1][tid.y][tid.z] + 2.0f * oldVal + ndx[tid.x + 1][tid.y][tid.z],
		//	ndx[tid.x][tid.y - 1][tid.z] + 2.0f * oldVal + ndx[tid.x][tid.y + 1][tid.z],
		//	ndx[tid.x][tid.y][tid.z - 1] + 2.0f * oldVal + ndx[tid.x][tid.y][tid.z + 1]
		//);


		

	}

	/////////// Compute

	float oldVal = ndx[tid.x][tid.y][tid.z];

	//http://janroman.dhis.org/finance/Numerical%20Methods/adi.pdf
	//float dt = 0.1f;
	int minDim = min(res.x, min(res.y, res.z));
	//float dt = 1.0f / (6.0f * minDim * D);
	//float3 dX = make_float3(1.0f / (res.x), 1.0f / (res.y), 1.0f / (res.z));
	float3 dX = make_float3(0.37e-6f);// , 1.0f / (res.y), 1.0f / (res.z));
	float3 dX2 = make_float3(dX.x*dX.x, dX.y*dX.y, dX.z*dX.z);


	float minD = max(params.zeroDiff, params.oneDiff);

	float dt = 1.0f / (2.0f * minD * (1.0f / dX2.x +  1.0f / dX2.y + 1.0f / dX2.z));
	
	//dt *= 1.0f / 10.0f;
	float3 v = D * make_float3(dt / dX2.x , dt / dX2.y, dt / dX2.z);

	//D(dt / 1 + dt / 1 + dt / 1) <= 1/2 >>> dt <= 1/6*D
	//3 * dt / (1/64)^2 <= 1/2 >>> dt <= 1/6 * 1/4096 >>> dt <= 1 / 24576*D 
	//D * dt * (1 / resx^2 + 1 / resy^2 + 1 / resz^2) <= 1/2
		//>>> dt <= 1/2 * 1/D * 1/ sum(dX.x^2)


//	if (vox.x < 2 && vox.y == 10 && vox.z == 10)
	//	printf("x %d, dt %.9f, vsum %.9f, val: %f dx: %f\n", vox.x ,dt , v.x+v.y+v.z, oldVal, oldVal - ndx[tid.x - 1][tid.y][tid.z]);

	
	float3 d2 = make_float3(
		ndx[tid.x - 1][tid.y][tid.z] + 2.0f * oldVal + ndx[tid.x + 1][tid.y][tid.z],
		ndx[tid.x][tid.y - 1][tid.z] + 2.0f * oldVal + ndx[tid.x][tid.y + 1][tid.z],
		ndx[tid.x][tid.y][tid.z - 1] + 2.0f * oldVal + ndx[tid.x][tid.y][tid.z + 1]
	);

	float newVal = oldVal +  (v.x * d2.x + v.y * d2.y + v.z * d2.z);
	
	//newVal = oldVal + v.x;
		

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







__global__ void kernelSubtract(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B) {

	VOLUME_VOX_GUARD(res);

	float Aval, Bval;
	surf3Dread(&Aval, A, vox.x * sizeof(float), vox.y, vox.z);
	surf3Dread(&Bval, B, vox.x * sizeof(float), vox.y, vox.z);

	float newVal = Bval - Aval;
	
	surf3Dwrite(newVal, A, vox.x * sizeof(float), vox.y, vox.z);
}

void launchSubtractKernel(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B) {
	uint3 block = make_uint3(8, 8, 8);
	uint3 numBlocks = make_uint3(
		(res.x / block.x) + 1,
		(res.y / block.y) + 1,
		(res.z / block.z) + 1
	);

	kernelSubtract << <numBlocks, block >> > (res, A, B);

}



//template <typename T, unsigned int blockSize>
//__global__ void reduce(T *g_idata, T *g_odata, unsigned int n)
//{
//	extern __shared__ int sdata[];
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
//	unsigned int gridSize = blockSize * 2 * gridDim.x;
//	sdata[tid] = 0;
//
//	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
//	__syncthreads();
//
//	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//	if (tid < 32) {
//		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}

__host__ __device__ uint3 ind2sub(uint3 res, uint i) {
	uint x = i % res.x;
	uint tmp = ((i - x) / res.x);
	uint y = tmp % res.y;
	uint z = (tmp - y) / res.y;
	return make_uint3(
		x,y,z
	);
}


template <typename T, unsigned int blockSize, bool toSurface>
__global__ void reduce3D(uint3 res, cudaSurfaceObject_t data, T * finalData, unsigned int n, uint3 offset)
{
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n) { 
		uint3 voxi = ind2sub(res, i);
		uint3 voxip = ind2sub(res, i + blockSize); 

		T vali, valip = 0;
		surf3Dread(&vali, data, voxi.x * sizeof(T), voxi.y, voxi.z);
		if(voxip.x < res.x && voxip.y < res.y && voxip.z < res.z)
			surf3Dread(&valip, data, voxip.x * sizeof(T), voxip.y, voxip.z);	
				
	
		sdata[tid] += (vali + valip);

		i += gridSize; 
	}
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) {
		unsigned int o = blockIdx.x;	
		//Either copy to surface
		if (toSurface) {						
			uint3 voxo = ind2sub(res,o);
			surf3Dwrite(sdata[0], data, voxo.x * sizeof(T), voxo.y, voxo.z);				
		}
		//Or final 1D array
		else {
			finalData[o] = sdata[0];						
		}	
	}
	
}



float launchReduceSumKernel(uint3 res, cudaSurfaceObject_t surf) {
		

	const uint finalSizeMax = 512;	
	const uint blockSize = 512;
	const uint3 block = make_uint3(blockSize,1,1);	
	uint n = res.x * res.y * res.z;

	//uint finalSizeMax = ((res.x * res.y * res.z) / blockSize) / 2;


	float * deviceResult = nullptr;
	cudaMalloc(&deviceResult, finalSizeMax * sizeof(float));
	cudaMemset(deviceResult, 0, finalSizeMax * sizeof(float));
	

	while (n > finalSizeMax) {
		uint3 numBlocks = make_uint3(
			(n / block.x) / 2 , 1, 1
		);		

		//If not final stage of reduction -> reduce into  surface
		if (numBlocks.x > finalSizeMax) {
			reduce3D<float, blockSize, true>
				<<<numBlocks, block, blockSize * sizeof(float)>>> (
					res, surf, nullptr, n, make_uint3(0)
					);
		}
		else {
			reduce3D<float, blockSize, false>
				<<<numBlocks, block, blockSize * sizeof(float)>>> (
					res, surf, deviceResult, n, make_uint3(0)
					);
		
		}

		//New N
		n = numBlocks.x;
	}


	float * hostResult = new float[finalSizeMax];
	cudaMemcpy(hostResult, deviceResult, finalSizeMax * sizeof(float), cudaMemcpyDeviceToHost);


	float result = 0.0f;
	for (auto i = 0; i < finalSizeMax; i++) {
		result += hostResult[i];
	}

	cudaFree(deviceResult);
	delete[] hostResult;


	return result;

}




//////////////////////////

template <Dir dir>
__global__ void sliceReduce(uint3 res, cudaSurfaceObject_t surf, float *output) {
		
	const uint3 tid = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);
	const uint tidLin = tid.x + tid.y * blockDim.x; 

	const uint3 vox = make_uint3(
		blockIdx.x * blockDim.x, //in slice x
		blockIdx.y * blockDim.y, //in slice y
		blockIdx.z * blockDim.z  //slice
	) + tid;

	{
		/*uint blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
		if(blockIndex == 0 || blockIndex == 127)
			printf("block: %d %d %d - > %d, vox: %d %d %d | blockDim: %d %d %d\n", 
				blockIdx.x, blockIdx.y, blockIdx.z,
				blockIndex, 
				vox.x, vox.y, vox.z,
				blockDim.x, blockDim.y, blockDim.z
			);*/
	}

	extern __shared__ float s[];

	uint3 texVox;
	if (dir == Z_POS || dir == Z_NEG)
		texVox = vox;
	else if (dir == Y_POS || dir == Y_POS)
		texVox = make_uint3(vox.x, vox.z, vox.y);
	else 
		texVox = make_uint3(vox.y, vox.z, vox.x);

	s[tidLin] = 0.0f;


	bool valid = true;
	if (texVox.x >= res.x || texVox.y > res.y || texVox.z >= res.z) {
		valid = false;
	}

	if(valid)
		surf3Dread(&s[tidLin], surf, texVox.x * sizeof(float), texVox.y, texVox.z);

	__syncthreads();

	
	//For now let tid==0 do all the work
	if (tidLin == 0) {
		float blockResult = 0.0f;
		uint perBlock = blockDim.x * blockDim.y;
		for (uint i = 0; i < perBlock; i++) {
			blockResult += s[i];
		}

		uint blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
		output[blockIndex] = blockResult;

	}

	
}


float launchReduceSumSlice(uint3 res, cudaSurfaceObject_t surf, Dir dir, void * output){
	



	const uint3 blockSize = make_uint3(16,16,1);
	const uint dirIndex = getDirIndex(dir);

	uint *resArr = (uint*)&res;
	uint3 resRotated = make_uint3(
		resArr[(dirIndex + 1) % 3], 
		resArr[(dirIndex + 2) % 3], 
		resArr[dirIndex]
	);

	//uint3 resRotated = make_uint3(864, 864, 289);

	const uint3 grid = roundDiv(resRotated, blockSize);

	const uint size = grid.x * grid.y * grid.z;

	float * deviceResult = nullptr;
	cudaMalloc(&deviceResult, size * sizeof(float));
	cudaMemset(deviceResult, 0, size * sizeof(float));

	if(dirIndex == 0)
		sliceReduce<X_POS> << <grid, blockSize, blockSize.x * blockSize.y * sizeof(float) >> >
			(res, surf, deviceResult);
	else if(dirIndex == 1)
		sliceReduce<Y_POS> << <grid, blockSize, blockSize.x * blockSize.y * sizeof(float) >> >
		(res, surf, deviceResult);
	else if (dirIndex == 2)
		sliceReduce<Z_POS> << <grid, blockSize, blockSize.x * blockSize.y * sizeof(float) >> >
		(res, surf, deviceResult);

		 

	float * hostResult = new float[size];
	cudaMemcpy(hostResult, deviceResult, size * sizeof(float), cudaMemcpyDeviceToHost);


	for (auto sliceID = 0; sliceID < grid.z; sliceID++) {

		((float *)output)[sliceID] = 0.0f;
		for (auto k = 0; k < grid.x * grid.y; k++) {
			((float *)output)[sliceID] += hostResult[grid.x * grid.y * sliceID + k];
		}

	}

	cudaFree(deviceResult);
	delete[] hostResult;





	//uint3 grid = make_uint3(resRotated.x / blockSize.x, roundDiv(resRotated.y, blockSize.y), resRotated.z / blockSize.z);	
	/*uint finalSizeMax = 1;
	while (sliceNum > finalSizeMax) 
		finalSizeMax *= 2;
	
	

	float * deviceResult = nullptr;
	cudaMalloc(&deviceResult, finalSizeMax * sizeof(float));
	cudaMemset(deviceResult, 0, finalSizeMax * sizeof(float));




	cudaFree(deviceResult);*/

	return 0;

}








/*
	Surface & buffer reduction, templated	
*/


template <typename T>
__device__ void opSum(volatile T & a, T b) {
	a += b;
}

template <typename T>
__device__ void opMin(volatile T & a, T b) {
	if (b < a) a = b;
}
template <typename T>
__device__ void opMax(volatile T & a, T  b) {
	if (b > a) a = b;
}



template <typename T>
using ReduceOp = void(*)(
	volatile T & a, T b
	);

template <typename T>
__device__ void opSquare(T & a) {
	a *= a;
}

template <typename T>
__device__ void opIdentity(T & a) {
	//nothing
}

template <typename T>
using PreReduceOp = void(*)(
	T & a
	);

template <typename T, unsigned int blockSize, ReduceOp<T> _op, PreReduceOp<T> _preOp = opIdentity<T>>
__global__ void reduce3DSurfaceToBuffer(uint3 res, cudaSurfaceObject_t surf, T * reducedData, size_t n)
{
	extern __shared__ __align__(sizeof(T)) volatile unsigned char my_smem[];
	volatile T *sdata = reinterpret_cast<volatile T *>(my_smem);

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = T(0);

	while (i < n) {
		const uint3 voxi = ind2sub(res, i);
		const uint3 voxip = ind2sub(res, i + blockSize);

		if (voxi.x < res.x && voxi.y < res.y && voxi.z < res.z) {
			
			/*if (threadIdx.x == 32 && threadIdx.y == 0 && threadIdx.z == 0) {
				if (blockIdx.x == 16 && blockIdx.y == 0 && blockIdx.z == 0) {
					printf("%d: %d %d %d ... %d %d %d\n",i, res.x, res.y, res.z , voxi.x, voxi.y, voxi.z);
				}
			}*/

			T vali = T(0);
			vali = read<T>(surf, voxi);
			_preOp(vali);
			_op(sdata[tid], vali);
		}

		if (i + blockSize < n && voxip.x < res.x && voxip.y < res.y && voxip.z < res.z) {
			T valip = T(0);
			valip = read<T>(surf, voxip);
			_preOp(valip);
			_op(sdata[tid], valip);
		}		

		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { _op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { _op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { _op(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) _op(sdata[tid], sdata[tid + 32]);
		if (blockSize >= 32) _op(sdata[tid], sdata[tid + 16]);
		if (blockSize >= 16) _op(sdata[tid], sdata[tid + 8]);
		if (blockSize >= 8) _op(sdata[tid], sdata[tid + 4]);
		if (blockSize >= 4) _op(sdata[tid], sdata[tid + 2]);
		if (blockSize >= 2) _op(sdata[tid], sdata[tid + 1]);
	}

	if (tid == 0) {		
		reducedData[blockIdx.x] = sdata[0];
	}

}


template <typename T, unsigned int blockSize, ReduceOp<T> _op>
__global__ void reduceBuffer(T * buffer, size_t n)
{
	extern __shared__ __align__(sizeof(T)) volatile unsigned char my_smem[];
	volatile T *sdata = reinterpret_cast<volatile T *>(my_smem);
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n) {
		_op(sdata[tid], buffer[i]);

		if(i + blockSize < n)
			_op(sdata[tid], buffer[i + blockSize]);

		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { _op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { _op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { _op(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) _op(sdata[tid], sdata[tid + 32]);
		if (blockSize >= 32) _op(sdata[tid], sdata[tid + 16]);
		if (blockSize >= 16) _op(sdata[tid], sdata[tid + 8]);
		if (blockSize >= 8) _op(sdata[tid], sdata[tid + 4]);
		if (blockSize >= 4) _op(sdata[tid], sdata[tid + 2]);
		if (blockSize >= 2) _op(sdata[tid], sdata[tid + 1]);
	}

	if (tid == 0) {
		buffer[blockIdx.x] = sdata[0];
	}

}



//AuxBuffer -> surf total n / 512
void launchReduceKernel(	
	PrimitiveType type, 
	ReduceOpType opType,
	uint3 res, 
	cudaSurfaceObject_t surf, 
	void * auxBufferGPU,
	void * auxBufferCPU,
	void * result	
) {

	const uint blockSize = VOLUME_REDUCTION_BLOCKSIZE;
	const uint sharedSize = primitiveSizeof(type) * blockSize;
	
	const uint3 block = make_uint3(blockSize, 1, 1);
	const uint finalSizeMax = VOLUME_REDUCTION_BLOCKSIZE;
	const size_t initialN = res.x * res.y * res.z;
	
	size_t n = initialN;
		
	/*
		Reduce from surface to auxiliar buffer
	*/
	{		
		uint3 numBlocks = make_uint3(
			(n / block.x) / 2, 1, 1
		);
		if (numBlocks.x == 0)
			numBlocks.x = 1;

		if (type == TYPE_FLOAT) {
			if (opType == REDUCE_OP_SQUARESUM)
				reduce3DSurfaceToBuffer<float, blockSize, opSum, opSquare> << <numBlocks, block, sharedSize>> > (
					res, surf, (float*)auxBufferGPU, n
					);						
		}
		else if (type == TYPE_DOUBLE) {
			if (opType == REDUCE_OP_SQUARESUM) {
				reduce3DSurfaceToBuffer<double, blockSize, opSum, opSquare> << <numBlocks, block, sharedSize >> > (
					res, surf, (double*)auxBufferGPU, n
					);
			}			

		}

		n = numBlocks.x;
	}


	/*
		Further reduce in buffer
	*/
	while (n > finalSizeMax) {
		const uint blockSize = VOLUME_REDUCTION_BLOCKSIZE;
		const uint3 block = make_uint3(blockSize, 1, 1);
		uint3 numBlocks = make_uint3(
				(n / block.x) / 2, 1, 1
			);

		if (type == TYPE_FLOAT) {
			if (opType == REDUCE_OP_SQUARESUM)
				reduceBuffer<float, blockSize, opSum><<<numBlocks,block, sharedSize>>>((float*)auxBufferGPU, n);				
		}
		if (type == TYPE_DOUBLE) {
			if (opType == REDUCE_OP_SQUARESUM)
				reduceBuffer<double, blockSize, opSum> << <numBlocks, block, sharedSize >> >((double*)auxBufferGPU, n);
		}


		n = numBlocks.x;
	}
	

	
	cudaMemcpy(auxBufferCPU, auxBufferGPU, primitiveSizeof(type) * n, cudaMemcpyDeviceToHost);

	/*
		Sum last array on CPU
	*/
	if (type == TYPE_FLOAT) {	
		*((float*)result) = 0.0f;
		
		for (auto i = 0; i < n; i++) {
			//printf("%f\n", ((float*)auxBufferCPU)[i]);			
			*((float*)result) += ((float*)auxBufferCPU)[i];
		}
	}
	else if (type == TYPE_DOUBLE) {	
		*((double*)result) = 0.0;
		for (auto i = 0; i < n; i++) {
			*((double*)result) += ((double*)auxBufferCPU)[i];
		}		
	}

	
	char b;
	b = 0;

	return;


	//float * deviceResult = nullptr;
	//cudaMalloc(&deviceResult, finalSizeMax * sizeof(float));
	//cudaMemset(deviceResult, 0, finalSizeMax * sizeof(float));


	//while (n > finalSizeMax) {
	//	uint3 numBlocks = make_uint3(
	//		(n / block.x) / 2, 1, 1
	//	);

	//	//If not final stage of reduction -> reduce into  surface
	//	if (numBlocks.x > finalSizeMax) {
	//		reduce3D<float, blockSize, true>
	//			<< <numBlocks, block, blockSize * sizeof(float) >> > (
	//				res, surf, nullptr, n, make_uint3(0)
	//				);
	//	}
	//	else {
	//		reduce3D<float, blockSize, false>
	//			<< <numBlocks, block, blockSize * sizeof(float) >> > (
	//				res, surf, deviceResult, n, make_uint3(0)
	//				);

	//	}

	//	//New N
	//	n = numBlocks.x;
	//}


	//float * hostResult = new float[finalSizeMax];
	//cudaMemcpy(hostResult, deviceResult, finalSizeMax * sizeof(float), cudaMemcpyDeviceToHost);


	//float result = 0.0f;
	//for (auto i = 0; i < finalSizeMax; i++) {
	//	result += hostResult[i];
	//}

	//cudaFree(deviceResult);
	//delete[] hostResult;


	//return result;

}

