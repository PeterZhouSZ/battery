#include "MultigridGPUNew.cuh"
#include "Volume.cuh"


#include <stdio.h>

__constant__ MultigridGPUParams MG;
static MultigridGPUParams MGHost;


cudaError_t _commitGPUParamsImpl(const MultigridGPUParams & mgHost) {
	MGHost = mgHost;	

	void * MGDev;
	cudaError_t res = cudaGetSymbolAddress((void **)&MGDev, MG);

	return cudaMemcpy(MGDev, &MGHost, sizeof(MultigridGPUParams), cudaMemcpyHostToDevice);
	//return cudaMemcpyToSymbol(&MG, &MGHost, sizeof(MultigridGPUParams), 0, cudaMemcpyHostToDevice);
}



template <typename T>
__device__ T convertMask(cudaSurfaceObject_t mask, uint3 vox,  T d0, T d1) {
	uchar bin = (read<uchar>(mask, vox) / 255); //0 for d0, 1 for d1
	return  (1 - bin) * d0 + bin * d1;
}

template <typename T>
__global__ void MGPrepareSystemKernel(int level, cudaSurfaceObject_t mask, double d0, double d1, double3 cellDim, Dir dir) {

	

	uint3 & res = MG.levels[level].dim;	
	VOLUME_VOX_GUARD(res);
	size_t total = res.x*res.y*res.z;
	
	//printf("lv %d, res: %d %d %d\n", MG.numLevels, res.x, res.y, res.z);
	

	const int dirPrimary = _getDirIndex(dir);
	const uint primaryRes = ((uint*)&res)[dirPrimary];
	
	T * F = ((T*)MG.levels[level].f);
	T * X = ((T*)MG.levels[level].x);

	T * A = ((T*)MG.levels[level].A.data);
	int * Arow = ((int*)MG.levels[level].A.row);
	int * Acol = ((int*)MG.levels[level].A.col);


	const T highConc = T(1.0);
	const T lowConc = T(0.0);
	const T concetrationBegin = (_getDirSgn(dir) == 1) ? highConc : lowConc;
	const T concetrationEnd = (_getDirSgn(dir) == 1) ? lowConc : highConc;

	
	const T faceArea[3] = { cellDim.y * cellDim.z,cellDim.x * cellDim.z,cellDim.x * cellDim.y };

	const size_t i = _linearIndex(res, vox);
	const size_t rowI = i * 7;
		

	

	T coeffs[7];
	bool useInMatrix[7];
	int cols[7];

	const Dir orderedDir[7] = {
		Z_NEG,
		Y_NEG,
		X_NEG,
		DIR_NONE,
		X_POS,
		Y_POS,
		Z_POS
	};


	{		

		T Di = convertMask(mask, vox, d0, d1);	

		T Dneg[3] = {
			(convertMask(mask, clampedVox(res, vox, X_NEG), d0, d1) + Di) * T(0.5),
			(convertMask(mask, clampedVox(res, vox, Y_NEG), d0, d1) + Di) * T(0.5),
			(convertMask(mask, clampedVox(res, vox, Z_NEG), d0, d1) + Di) * T(0.5)
		};

		T Dpos[3] = {
			(convertMask(mask, clampedVox(res, vox, X_POS), d0, d1) + Di) * T(0.5),
			(convertMask(mask, clampedVox(res, vox, Y_POS), d0, d1) + Di) * T(0.5),
			(convertMask(mask, clampedVox(res, vox, Z_POS), d0, d1) + Di) * T(0.5)
		};

		
		//Diagonal
		coeffs[3] = T(0);
		useInMatrix[3] = true;
		
		#pragma unroll
		for (uint j = 0; j < 7; j++) {
			const Dir d = orderedDir[j];
			const uint k = _getDirIndex(d);
			const int sgn = _getDirSgn(d);
			cols[j] = _linearIndex(res,vox + make_uint3(dirVec(d)));

			useInMatrix[j] = true;
			if (d == DIR_NONE) continue;

			T cellDist[3] = { cellDim.x,cellDim.y,cellDim.z };
			if ((_at<uint>(vox, k) == 0 && sgn == -1) ||
				(_at<uint>(vox, k) == _at<uint>(res, k) - 1 && sgn == 1)
				) {
				cellDist[k] = _at<double>(cellDim,k) * T(0.5);
				useInMatrix[j] = false;
			}

			const T Dface = (sgn == -1) ? Dneg[k] : Dpos[k];
			coeffs[j] = (Dface * faceArea[k]) / cellDist[k];

			//Subtract from diagonal
			if (useInMatrix[j] || k == dirPrimary)
				coeffs[3] -= coeffs[j];
		}
	}
	
	

	
	//Right hand side	
	T rhs = T(0);
	{
		if (_at<uint>(vox, dirPrimary) == 0) {
			Dir dir = _getDir(dirPrimary, -1);
			rhs -= coeffs[dir] * concetrationBegin;
		}
		else if (_at<uint>(vox, dirPrimary) == primaryRes - 1) {
			Dir dir = _getDir(dirPrimary, 1);
			rhs -= coeffs[dir] * concetrationEnd;
		}
	}

	//X initial guess
	T xGuess = T(0);
	if (_getDirSgn(dir) == 1)
		xGuess = 1.0f - (_at<uint>(vox, dirPrimary) / T(primaryRes + 1));
	else
		xGuess = (_at<uint>(vox, dirPrimary) / T(primaryRes + 1));
	
	//Assign columns (without collision) to unused coefficients (since there are 7 allocated)
	int nonzeroI = 0;
	for (uint j = 0; j < 7; j++) {
		if (!useInMatrix[j]) {
			coeffs[j] = 0.0;
			cols[j] = (cols[3] + 2 + total + nonzeroI) % total;
			nonzeroI++;
		}
	}

	//Bubble sort	
	for (uint j = 0; j < 7 - 1; j++) {		
		for (uint k = 0; k < 7 - 1 - j; k++) {
			if (cols[k] > cols[k + 1]) {
				int tmpc = cols[k + 1];
				cols[k + 1] = cols[k];
				cols[k] = tmpc;

				T tmpco = coeffs[k + 1];
				coeffs[k + 1] = coeffs[k];
				coeffs[k] = tmpco;
			}
		}
	}

	//Write to X
	X[i] = xGuess;

	//Write to F
	F[i] = rhs;


	//Write row pointer
	Arow[i] = i * 7; 	

	//Write col pointer
	Acol[rowI + 0] = cols[0];
	Acol[rowI + 1] = cols[1];
	Acol[rowI + 2] = cols[2];
	Acol[rowI + 3] = cols[3];
	Acol[rowI + 4] = cols[4];
	Acol[rowI + 5] = cols[5];
	Acol[rowI + 6] = cols[6];

	//Write coeffs
	A[rowI + 0] = coeffs[0];
	A[rowI + 1] = coeffs[1];
	A[rowI + 2] = coeffs[2];
	A[rowI + 3] = coeffs[3];
	A[rowI + 4] = coeffs[4];
	A[rowI + 5] = coeffs[5];
	A[rowI + 6] = coeffs[6];

	return; 
	
}

void MGPrepareSystem(int level, cudaSurfaceObject_t mask, double d0, double d1, double3 cellDim, Dir dir) {

	BLOCKS3D(8, MGHost.levels[level].dim);

	if (MGHost.type == TYPE_FLOAT) {
		MGPrepareSystemKernel<float> << < numBlocks, block >> >(level, mask,d0,d1,cellDim,dir);
	}
	else if (MGHost.type == TYPE_DOUBLE) {
		MGPrepareSystemKernel<double> << < numBlocks, block >> > (level, mask, d0, d1, cellDim, dir);
	}

	

}