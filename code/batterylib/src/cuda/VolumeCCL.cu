#include "VolumeCCL.cuh"
#include <stdio.h>
#include <assert.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <set>

/*
	Based on
	Ohira, N. (2018). Memory-efficient 3D connected component labeling with parallel computing. Signal, Image and Video Processing, 12(3), 429-436.
*/


struct VolumeCCL_Params {
	size_t spanCount;

	//Input resolution	
	uint3 res;

	//Resolution of span/label volumes
	uint3 spanRes;	
	uint2 * spanVolume;
	uint * labelVolume;

	uint * spanCountPlane;

	bool * hasChanged;
};


__global__ void __countSpansKernel(CUDA_Volume input,  uint * result) {

	const uint3 resultRes = make_uint3(1, input.res.y, input.res.z);
	VOLUME_IVOX_GUARD(resultRes);

	using T = uchar;

	uint spanSum = 0;
	int3 curPos = ivox;

	T prevVal = T(0);

	bool open = false;
	for (int i = 0; i < input.res.x; i++) {
		T newVal = read<T>(input.surf, curPos);
		

		if (prevVal != newVal) {
			open = !open;
			if (open) {
				spanSum++;				
			}
		}		
		prevVal = newVal;
		curPos.x++;
	}

	size_t linIndex = _linearIndex(resultRes, ivox);

	result[linIndex] = spanSum;

}


/*
	Detect spans again and assign them to span and label matrices
*/
__global__ void __buildMatricesKernel(
	CUDA_Volume input,
	VolumeCCL_Params p
) {
	const uint3 planeRes = make_uint3(1, input.res.y, input.res.z);
	VOLUME_IVOX_GUARD(planeRes);

	using T = uchar;

	uint currentSpanIndex = 0;
	int3 curPos = ivox;

	T prevVal = T(0);
	uint spanBegin = 0;

	bool open = false;
	for (int i = 0; i < input.res.x; i++) {
		T newVal = read<T>(input.surf, curPos);

		if (prevVal != newVal) {
			if (!open) {				
				spanBegin = i;								
			}
			else {		
				size_t index = _linearIndex(p.spanRes, make_int3(currentSpanIndex, ivox.y, ivox.z));
				p.spanVolume[index] = make_uint2(spanBegin, i - 1);				
				p.labelVolume[index] = _linearIndex(input.res, make_uint3(spanBegin, ivox.y, ivox.z));
				currentSpanIndex++;
			}
			open = !open;
		}
		prevVal = newVal;
		curPos.x++;
	}

	//Close last one if open
	if (open) {
		size_t index = _linearIndex(p.spanRes, make_int3(currentSpanIndex, ivox.y, ivox.z));
		p.spanVolume[index] = make_uint2(spanBegin, input.res.x - 1);
		p.labelVolume[index] = _linearIndex(input.res, make_uint3(spanBegin, ivox.y, ivox.z));
	}

}


inline __device__ bool spanOverlap(uint2 a, uint2 b) {
	return a.x <= b.y && b.x <= a.y;
}

inline __device__ uint labelEquivalence(VolumeCCL_Params & p, uint index, int3 ivox) {

	
	
	uint indexPrev = 0;	
	do {
		indexPrev = index;	

		//Find pos of label
		uint3 posOrig = posFromLinear(p.res, index);
		uint3 posLabel = make_uint3(0, posOrig.y, posOrig.z);

		const uint rowSpanCount = p.spanCountPlane[_linearIndex(make_uint3(1, p.res.y, p.res.z), posLabel)];
		for (int i = 0; i < rowSpanCount; i++) {			
			uint2 span = p.spanVolume[_linearIndex(p.spanRes, posLabel)];
			if (posOrig.x >= span.x && posOrig.x <= span.y)
				break;		
			posLabel.x++;
		}
		index = p.labelVolume[_linearIndex(p.spanRes, posLabel)];
		
	}
	while(index != indexPrev);

	return index;
}

//input surface not needed
__global__ void __updateContinuityKernel(VolumeCCL_Params p) {
	const uint3 planeRes = make_uint3(1, p.res.y, p.res.z);
	VOLUME_IVOX_GUARD(planeRes);

	const uint rowSpanCount = p.spanCountPlane[_linearIndex(planeRes, ivox)];
	
	int3 curPosSpan = ivox;
	

	const int3 offsets[4] = {
		{ 0,-1,0 },
		{ 0,1,0 },
		{ 0,0,-1 },
		{ 0,0,1 },
	};

	int3 otherPosSpan[4] = {
		ivox + offsets[0], ivox + offsets[1], ivox + offsets[2], ivox + offsets[3]
	};



	const uint otherSpanCount[4] = {
		(ivox.y == 0) ?				0 : p.spanCountPlane[_linearIndex(planeRes, ivox + offsets[0])],
		(ivox.y == p.res.y - 1) ?	0 : p.spanCountPlane[_linearIndex(planeRes, ivox + offsets[1])],
		(ivox.z == 0) ?				0 : p.spanCountPlane[_linearIndex(planeRes, ivox + offsets[2])],
		(ivox.z == p.res.z - 1) ?	0 : p.spanCountPlane[_linearIndex(planeRes, ivox + offsets[3])]
	};

	for (int i = 0; i < rowSpanCount; i++) {
		uint2 thisSpan = p.spanVolume[_linearIndex(p.spanRes, curPosSpan)];		
		uint tempLabel = p.labelVolume[_linearIndex(p.spanRes, curPosSpan)];
		

		

		#pragma unroll
		for (int k = 0; k < 4; k++) {
			

			if (k == 0 && ivox.y == 0) continue;
			if (k == 1 && ivox.y == p.res.y - 1) continue;
			if (k == 2 && ivox.z == 0) continue;
			if (k == 3 && ivox.z == p.res.z - 1) continue;
	
			
			while(otherPosSpan[k].x < otherSpanCount[k]){
				uint2 otherSpan = p.spanVolume[_linearIndex(p.spanRes, otherPosSpan[k])];

				

				if (otherSpan.x > thisSpan.y) break;							

				if (spanOverlap(thisSpan, otherSpan)) {										
					uint * thisLabelPtr = p.labelVolume + _linearIndex(p.spanRes, curPosSpan);
					uint * otherLabelPtr = p.labelVolume + _linearIndex(p.spanRes, otherPosSpan[k]);

					uint thisLabel = *thisLabelPtr;
					uint otherLabel = *otherLabelPtr;
					
					if (thisLabel < otherLabel) {
						atomicMin(otherLabelPtr, thisLabel);
						*p.hasChanged = true;
					}
					else if(otherLabel < thisLabel) {
						atomicMin(thisLabelPtr, otherLabel);
						*p.hasChanged = true;
					}				
					
				}

				otherPosSpan[k].x++;				
			}
		}	

		curPosSpan.x++;
		

	}


	curPosSpan = ivox;	
	size_t index0 = _linearIndex(p.res, ivox);	

	for (int i = 0; i < rowSpanCount; i++) {
		uint index = p.labelVolume[_linearIndex(p.spanRes, curPosSpan)];


		p.labelVolume[_linearIndex(p.spanRes, curPosSpan)] = labelEquivalence(p, index, ivox);

		curPosSpan.x++;		
	}


}



__global__ void __labelOutputKernelUchar(
	VolumeCCL_Params p,
	CUDA_Volume output,
	uint label
) {
	const uint3 planeRes = make_uint3(1, p.res.y, p.res.z);
	VOLUME_IVOX_GUARD(planeRes);

	const uint rowSpanCount = p.spanCountPlane[_linearIndex(planeRes, ivox)];

	int3 posSpan = ivox;

	for (int i = 0; i < rowSpanCount; i++) {

		const uint2 thisSpan = p.spanVolume[_linearIndex(p.spanRes, posSpan)];
		const uint thisLabel = p.labelVolume[_linearIndex(p.spanRes, posSpan)];

		if (thisLabel != label) {
			posSpan.x++;
			continue;
		}
		

		for (int k = thisSpan.x; k <= thisSpan.y; k++) {
			const int3 pos = make_int3(k, ivox.y, ivox.z);
			write<uchar>(output.surf, pos, 255);
		}

		posSpan.x++;
	}

}


//#ifdef _DEBUG 
//#define DEBUG_CPU
//#endif

bool VolumeCCL_Compute(const CUDA_Volume & input, CUDA_Volume & output, uint label)
{
	assert(input.type == TYPE_UCHAR);
	//assert(output.type == TYPE_UINT);
	
	VolumeCCL_Params p;	
	p.res = input.res;

	thrust::device_vector<uint> sumPlane(input.res.y * input.res.z);
	p.spanCountPlane = sumPlane.data().get();

	//Count spands
	{
		BLOCKS3D_INT3(1, 8, 8, make_uint3(1, input.res.y, input.res.z));
		//Summing in ascending X direction
		
		__countSpansKernel << < numBlocks, block >> > (input, p.spanCountPlane);
		uint maxSpanCount = thrust::reduce(sumPlane.begin(), sumPlane.end(), 0, thrust::maximum<uint>());


		p.spanCount = maxSpanCount;
		p.spanRes = make_uint3(p.spanCount, p.res.y, p.res.z);		
		
	}
#ifdef DEBUG_CPU
	thrust::host_vector<uint> hostSum = sumPlane;
	uint * dataSum = hostSum.data();
#endif
	

	{
		

		thrust::device_vector<uint2> spanMatrix(p.spanRes.x * p.spanRes.y * p.spanRes.z);
		thrust::device_vector<uint> labelMatrix(p.spanRes.x * p.spanRes.y * p.spanRes.z);
		p.spanVolume = spanMatrix.data().get();
		p.labelVolume = labelMatrix.data().get();

#ifdef DEBUG_CPU
		thrust::host_vector<uint2> hostSpan;
		thrust::host_vector<uint> hostLabel;

		cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 64);
#endif

		cudaMalloc(&p.hasChanged, 1);
		cudaMemset(p.hasChanged, 0, 1);

		{
			BLOCKS3D_INT3(1, 8, 8, make_uint3(1, input.res.y, input.res.z));
			__buildMatricesKernel << < numBlocks, block >> > (input, p);
		}

#ifdef DEBUG_CPU
		{
			hostSpan = spanMatrix;  uint2 * dataSpan = hostSpan.data();
			hostLabel = labelMatrix; uint * dataLabel = hostLabel.data();
			char b;
			b = 0;
		}

		
#endif

		bool hasChangedHost = false;
		int iteration = 0;
		do
		{
			BLOCKS3D_INT3(1, 8, 8, make_uint3(1, input.res.y, input.res.z));
			__updateContinuityKernel << < numBlocks, block >> >(p);
			cudaMemcpy(&hasChangedHost, p.hasChanged, 1, cudaMemcpyDeviceToHost);
			cudaMemset(p.hasChanged, 0, 1);

#ifdef DEBUG_CPU
			{
								
				hostSpan = spanMatrix;  uint2 * dataSpan = hostSpan.data();
				hostLabel = labelMatrix; uint * dataLabel = hostLabel.data();
				std::set<uint> uniqueLabels;

				for (auto i = 0; i < p.spanRes.x * p.spanRes.y * p.spanRes.z; i++) {
					uniqueLabels.insert(hostLabel[i]);
				}
				

				/*for (auto & s : uniqueLabels) {
					printf("%u, ", s);
				}
				printf("\n");*/
				char b;
				b = 0;

				iteration++;
			}
#endif

		} while (hasChangedHost);			
		


		cudaFree(p.hasChanged);
		


		//Reconstruct volume

		{
			assert(output.type == TYPE_UCHAR);
			BLOCKS3D_INT3(1, 8, 8, make_uint3(1, input.res.y, input.res.z));
			__labelOutputKernelUchar << < numBlocks, block >> > (p, output, label);
		}

		
		

	}

	


	return true;
}




