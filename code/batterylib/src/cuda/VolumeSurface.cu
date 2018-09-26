#include "VolumeSurface.cuh"
#include <assert.h>
#include <stdio.h>

#include "MCTable.cuh"

#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "../include/DataPtr.h"


__device__ __constant__ uchar const_numVertsTable[256];
__device__ __constant__ uint const_edgeTable[256];
__device__ __constant__ uchar const_triTable[256 * 16];



struct VoxelCornerVals {
	float v[8];
};

#define TRIANGLE_THREADS 32

bool commitMCConstants() {
	if (cudaMemcpyToSymbol(const_numVertsTable,	&numVertsTable,
		sizeof(uchar) * 256,0,
		cudaMemcpyHostToDevice
	) != cudaSuccess) return false;

	if (cudaMemcpyToSymbol(const_edgeTable, &edgeTable,
		sizeof(uint) * 256, 0,
		cudaMemcpyHostToDevice
	) != cudaSuccess) return false;

	if (cudaMemcpyToSymbol(const_triTable, &triTable,
		sizeof(uchar) * 256 * 16, 0,
		cudaMemcpyHostToDevice
	) != cudaSuccess) return false;

	return true;
}


inline __device__ VoxelCornerVals getVoxelCornerVals(
	cudaTextureObject_t tex,
	float3 pos,
	float3 voxelSize
) {
	VoxelCornerVals vals;
	//printf("Reading at pos %f %f %f\n", pos.x, pos.y, pos.z);
	vals.v[0] = tex3D<float>(tex, pos.x, pos.y, pos.z);
	vals.v[1] = tex3D<float>(tex, pos.x + voxelSize.x, pos.y, pos.z);
	vals.v[2] = tex3D<float>(tex, pos.x + voxelSize.x, pos.y + voxelSize.y, pos.z);
	vals.v[3] = tex3D<float>(tex, pos.x, pos.y + voxelSize.y, pos.z);

	vals.v[4] = tex3D<float>(tex, pos.x, pos.y, pos.z + voxelSize.z);
	vals.v[5] = tex3D<float>(tex, pos.x + voxelSize.x, pos.y, pos.z + voxelSize.z);
	vals.v[6] = tex3D<float>(tex, pos.x + voxelSize.x, pos.y + voxelSize.y, pos.z + voxelSize.z);
	vals.v[7] = tex3D<float>(tex, pos.x, pos.y + voxelSize.y, pos.z + voxelSize.z);
	
	return vals;
}

template <bool gt = false>
inline __device__ uint getCubeIndex(const VoxelCornerVals & vals, float isoValue) {

	uint cubeindex = 0;
	if (!gt) {
		cubeindex = uint(vals.v[0] < isoValue);
		cubeindex += uint(vals.v[1] < isoValue) * 2;
		cubeindex += uint(vals.v[2] < isoValue) * 4;
		cubeindex += uint(vals.v[3] < isoValue) * 8;
		cubeindex += uint(vals.v[4] < isoValue) * 16;
		cubeindex += uint(vals.v[5] < isoValue) * 32;
		cubeindex += uint(vals.v[6] < isoValue) * 64;
		cubeindex += uint(vals.v[7] < isoValue) * 128;
	}
	else {
		cubeindex = uint(vals.v[0] > isoValue);
		cubeindex += uint(vals.v[1] > isoValue) * 2;
		cubeindex += uint(vals.v[2] > isoValue) * 4;
		cubeindex += uint(vals.v[3] > isoValue) * 8;
		cubeindex += uint(vals.v[4] > isoValue) * 16;
		cubeindex += uint(vals.v[5] > isoValue) * 32;
		cubeindex += uint(vals.v[6] > isoValue) * 64;
		cubeindex += uint(vals.v[7] > isoValue) * 128;
	}
	return cubeindex;

}

__global__ void ___markVoxels(
	uint3 MCRes, 
	CUDA_Volume volume,
	uint * vertCount,
	uint * occupancy,
	float isoValue
) {

	VOLUME_IVOX_GUARD(MCRes);
	const size_t i = _linearIndex(MCRes, ivox);	
	

	const float3 voxelSize = make_float3(1.0f / MCRes.x, 1.0f / MCRes.y, 1.0f / MCRes.z);	
	const  float3 pos = make_float3(voxelSize.x * ivox.x, voxelSize.y * ivox.y, voxelSize.z * ivox.z);
	
	//Sample volume
	const VoxelCornerVals vals = getVoxelCornerVals(volume.tex, pos, voxelSize);

	//Calculate occupancy
	const uint cubeindex = getCubeIndex(vals, isoValue);	
	const uint numVerts = uint(const_numVertsTable[cubeindex]);

	/*if (ivox.x == 0 && ivox.y == 0 && ivox.z == 0) {
		printf("i %u, %f %f %f, |%f|%f|%f|%f||%f|%f|%f|%f| -> %u %u \n", uint(i), pos.x, pos.y, pos.z,
			vals.v[0], vals.v[1], vals.v[2], vals.v[3], vals.v[4], vals.v[5], vals.v[6], vals.v[7],
			cubeindex, numVerts
			);
	}*/

	vertCount[i] = numVerts;
	occupancy[i] = uchar(numVerts > 0);

}

__device__
float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	if (f1 == f0) {
		return p0;		
	}
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}


__device__
float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
	float3 edge0 = *v1 - *v0;
	float3 edge1 = *v2 - *v0;
	// note - it's faster to perform normalization in vertex shader rather than here

	return normalize(cross(edge0, edge1));
}



__global__ void ___generateTriangles(
	const uint3 MCRes,
	const CUDA_Volume volume,
	const uint * compacted,
	const uint * vertCountScan,	
	const size_t activeN,
	const float isoValue,	
	blib::CUDA_VBO::DefaultAttrib * vbo
) {

	uint blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	uint i = blockId * blockDim.x + threadIdx.x;
	
	if (i > activeN - 1) {
		return;		
	}

	uint voxelIndex = compacted[i];
	int3 vox = posFromLinear(make_int3(MCRes), voxelIndex);
	int3 ivox = vox;

	
	//Recompute corner vals
	const float3 voxelSize = make_float3(1.0f / MCRes.x, 1.0f / MCRes.y, 1.0f / MCRes.z);
	const float3 pos = make_float3(voxelSize.x * ivox.x, voxelSize.y * ivox.y, voxelSize.z * ivox.z);
	const VoxelCornerVals vals = getVoxelCornerVals(volume.tex, pos, voxelSize);
	const uint cubeindex = getCubeIndex(vals, isoValue);

	//Target position of the mesh
	const float3 targetVoxelSize = make_float3(2.0f / MCRes.x, 2.0f / MCRes.y, 2.0f / MCRes.z);
	const float3 targetPos = make_float3(targetVoxelSize.x * ivox.x, targetVoxelSize.y * ivox.y, targetVoxelSize.z * ivox.z) - make_float3(1.0f);
	

	float3 v[8];
	v[0] = targetPos;
	v[1] = targetPos + make_float3(targetVoxelSize.x, 0, 0);
	v[2] = targetPos + make_float3(targetVoxelSize.x, targetVoxelSize.y, 0);
	v[3] = targetPos + make_float3(0, targetVoxelSize.y, 0);
	v[4] = targetPos + make_float3(0, 0, targetVoxelSize.z);
	v[5] = targetPos + make_float3(targetVoxelSize.x, 0, targetVoxelSize.z);
	v[6] = targetPos + make_float3(targetVoxelSize.x, targetVoxelSize.y, targetVoxelSize.z);
	v[7] = targetPos + make_float3(0, targetVoxelSize.y, targetVoxelSize.z);

	
	//printf("%u, %f %f %f, %f %f %f | %f %f %f %f \n", voxelIndex, v[0].x, v[0].y, v[0].z, v[1].x, v[1].y, v[1].z, vals.v[0], vals.v[1], vals.v[2], vals.v[3]);

	__shared__ float3 vertlist[12 * TRIANGLE_THREADS];	

	vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], vals.v[0], vals.v[1]);
	//printf("%f | %f %f %f | )
	vertlist[TRIANGLE_THREADS + threadIdx.x] = vertexInterp(isoValue, v[1], v[2], vals.v[1], vals.v[2]);
	vertlist[(TRIANGLE_THREADS * 2) + threadIdx.x] = vertexInterp(isoValue, v[2], v[3], vals.v[2], vals.v[3]);
	vertlist[(TRIANGLE_THREADS * 3) + threadIdx.x] = vertexInterp(isoValue, v[3], v[0], vals.v[3], vals.v[0]);
	vertlist[(TRIANGLE_THREADS * 4) + threadIdx.x] = vertexInterp(isoValue, v[4], v[5], vals.v[4], vals.v[5]);
	vertlist[(TRIANGLE_THREADS * 5) + threadIdx.x] = vertexInterp(isoValue, v[5], v[6], vals.v[5], vals.v[6]);
	vertlist[(TRIANGLE_THREADS * 6) + threadIdx.x] = vertexInterp(isoValue, v[6], v[7], vals.v[6], vals.v[7]);
	vertlist[(TRIANGLE_THREADS * 7) + threadIdx.x] = vertexInterp(isoValue, v[7], v[4], vals.v[7], vals.v[4]);
	vertlist[(TRIANGLE_THREADS * 8) + threadIdx.x] = vertexInterp(isoValue, v[0], v[4], vals.v[0], vals.v[4]);
	vertlist[(TRIANGLE_THREADS * 9) + threadIdx.x] = vertexInterp(isoValue, v[1], v[5], vals.v[1], vals.v[5]);
	vertlist[(TRIANGLE_THREADS * 10) + threadIdx.x] = vertexInterp(isoValue, v[2], v[6], vals.v[2], vals.v[6]);
	vertlist[(TRIANGLE_THREADS * 11) + threadIdx.x] = vertexInterp(isoValue, v[3], v[7], vals.v[3], vals.v[7]);
	__syncthreads();


	
	uint numVerts = uint(const_numVertsTable[cubeindex]);

	//printf("%u (%d %d %d): %f %f %f -> %f %f %f | %u\n", voxelIndex, vox.x, vox.y, vox.z, pos.x, pos.y, pos.z, targetPos.x, targetPos.y, targetPos.z, numVerts);

	for (int j = 0; j < numVerts; j += 3)
	{
		uint index = vertCountScan[voxelIndex] + j;

		float3 *vert[3];
		uint edge;
		edge = uint(const_triTable[(cubeindex * 16) + j]);		
		//edge = const_triTable[]
		vert[0] = &vertlist[(edge*TRIANGLE_THREADS) + threadIdx.x];
		
		edge = uint(const_triTable[(cubeindex * 16) + j + 1]);
		vert[1] = &vertlist[(edge*TRIANGLE_THREADS) + threadIdx.x];
		
		edge = uint(const_triTable[(cubeindex * 16) + j + 2]);
		vert[2] = &vertlist[(edge*TRIANGLE_THREADS) + threadIdx.x];


		// calculate triangle surface normal
		//float3 n = calcNormal(v[0], v[1], v[2]);

		/*if (index < (maxVerts - 3))
		{*/

		float3 n = calcNormal(vert[0], vert[1], vert[2]);

		#pragma unroll
		for (int k = 0; k < 3; k++) {
			blib::CUDA_VBO::DefaultAttrib & att = vbo[index + k];


			
			att.pos[0] = (*vert[k]).x;
			att.pos[1] = (*vert[k]).y;
			att.pos[2] = (*vert[k]).z;

			//printf("%u %f %f %f\n", voxelIndex, att.pos[0], att.pos[1], att.pos[2]);

			att.normal[0] = n.x;
			att.normal[1] = n.y;
			att.normal[2] = n.z;

			//float3 c = make_float3(n.x+ 1.0f, n.y + 1.0f, n.z + 1.0f);
			const float3 c = make_float3(1.0f);		

			att.color[0] = c.x;
			att.color[1] = c.y;
			att.color[2] = c.z;
			att.color[3] = 1.0;
		
		}		
	}



}


__global__ void ___compactVoxels(
	const uint3 res,
	const uint * occupancy,
	const uint * occupancyScan,		
	uint * compacted
) {
	VOLUME_IVOX_GUARD(res);
	size_t i = _linearIndex(res, ivox);

	if (occupancy[i]) {
		compacted[occupancyScan[i]] = i;
	}
}

template <typename T>
T scanAndCount(T * input, T * scanned, size_t N) {
	thrust::exclusive_scan(thrust::device_ptr<T>(input),
		thrust::device_ptr<T>(input + N),
		thrust::device_ptr<T>(scanned));

	uint lastInput, lastScan;
	cudaMemcpy(&lastInput, (input + N - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(&lastScan, (scanned + N - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	return lastInput + lastScan;
}

//#define MC_DEBUG_CPU

void VolumeSurface_MarchingCubesMesh(const CUDA_Volume & input, uint3 MCRes, float isovalue, uint * vboOut, size_t * NvertsOut)
{
	

	

	
	assert(input.type == TYPE_UCHAR);

	commitMCConstants();

	const float isoValue = isovalue;

	size_t N = MCRes.x * MCRes.y * MCRes.z;

	if (*vboOut && *NvertsOut) {
		*NvertsOut = 0;
	}


	uint * vertCount;
	uint * vertCountScan;
	uint * occupancy;
	uint * occupancyScan;
	uint * compacted;

	cudaMalloc(&vertCount, N * sizeof(uint));
	cudaMalloc(&vertCountScan, N * sizeof(uint));
	cudaMalloc(&occupancy, N * sizeof(uint));
	cudaMalloc(&occupancyScan, N * sizeof(uint));
	cudaMalloc(&compacted, N * sizeof(uint));

	auto freeResouces = [=]{
		cudaFree(vertCount);
		cudaFree(vertCountScan);
		cudaFree(occupancy);
		cudaFree(occupancyScan);
		cudaFree(compacted);
	};


	//Find occupied voxels and number of triangles
	{
		BLOCKS3D(8, MCRes);
		___markVoxels<<< numBlocks, block>>>(
			MCRes, input, vertCount, occupancy, isoValue
		);
	}

	

	//Perform scan on occupancy -> counts number of occupied voxels
	uint activeN = scanAndCount(occupancy, occupancyScan, N);
	
	
	
	if (activeN == 0) {		
		freeResouces();
		return;
	}

	//Compact
	{
		BLOCKS3D(8, MCRes);
		___compactVoxels << < numBlocks, block >> > (MCRes, occupancy, occupancyScan, compacted);
	}

	//Scan vert count
	uint totalVerts = scanAndCount(vertCount, vertCountScan, N);	
	blib::CUDA_VBO cudaVBO = blib::createMappedVBO(totalVerts * sizeof(blib::CUDA_VBO::DefaultAttrib));


	{
		dim3 grid = dim3((activeN + TRIANGLE_THREADS - 1) / TRIANGLE_THREADS, 1, 1);
		while (grid.x > 65535) {
			grid.x /= 2;
			grid.y *= 2;
		}

		___generateTriangles<< <grid, TRIANGLE_THREADS >>>(
			MCRes,
			input, compacted, vertCountScan, activeN, isoValue,
			static_cast<blib::CUDA_VBO::DefaultAttrib *>(cudaVBO.getPtr())
			);

	}


#ifdef MC_DEBUG_CPU
	uint * vertCountCpu = new uint[N];
	uint * vertCountScanCpu = new uint[N];
	uint * occupancyCpu = new uint[N];
	uint * occupancyScanCpu = new uint[N];
	uint * compactedCpu = new uint[N];
	cudaMemcpy(vertCountCpu, vertCount, N * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(vertCountScanCpu, vertCountScan, N * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(occupancyCpu, occupancy, N * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(occupancyScanCpu, occupancyScan, N * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(compactedCpu, compacted, N * sizeof(uint), cudaMemcpyDeviceToHost);

	blib::CUDA_VBO::DefaultAttrib * vertsCpu = new blib::CUDA_VBO::DefaultAttrib[totalVerts];
	cudaVBO.retrieveTo(vertsCpu);


#endif


	*vboOut = cudaVBO.getVBO();
	*NvertsOut = totalVerts;
	
	freeResouces();	

}
