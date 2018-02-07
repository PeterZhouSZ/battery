#pragma once

#include "BatteryLibDef.h"
#include "Volume.h"

#include <cusparse.h>
#include <cusolverSp.h>


namespace blib {

	class DiffusionSolver {


	public:
		BLIB_EXPORT DiffusionSolver(bool verbose = true);
		BLIB_EXPORT ~DiffusionSolver();


		//Solves stable=state diffusion equation
		//subdim allows to select smaller subvolume - for testing
		//TODO: diffusion params
		BLIB_EXPORT bool solve(
			VolumeChannel & volChannel, 						
			VolumeChannel * outVolume = nullptr,
			ivec3 subdim = ivec3(INT_MAX)
		);

		BLIB_EXPORT bool solveWithoutParticles(
			VolumeChannel & volChannel,
			VolumeChannel * outVolume = nullptr,
			ivec3 subdim = ivec3(INT_MAX)
		);

		BLIB_EXPORT float tortuosityCPU(
			const VolumeChannel & mask,
			const VolumeChannel & concetration,
			Dir dir
		);

	private:
			
		
		bool _verbose;

		cusolverSpHandle_t _handle = nullptr;
		cusparseHandle_t _cusparseHandle = nullptr; // used in residual evaluation
		cudaStream_t _stream = nullptr;
		cusparseMatDescr_t _descrA = nullptr;

		size_t _N = 0;
		size_t _M = 0;
		size_t _nnz = 0;
		size_t _maxElemPerRow = 7;

		float * _deviceA = nullptr;
		int * _deviceRowPtr = nullptr;
		int * _deviceColInd = nullptr;

		float * _deviceB = nullptr;
		float * _deviceX = nullptr;

	};

}