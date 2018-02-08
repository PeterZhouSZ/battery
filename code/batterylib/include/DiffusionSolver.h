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
		//if dir is pos, 0 is high concetration, otherwise dim[dir]-1 is high
		BLIB_EXPORT bool solve(
			VolumeChannel & volChannel, 						
			VolumeChannel * outVolume,
			Dir dir,
			float d0,
			float d1,
			float tolerance = 1.0e-6f			
		);

		BLIB_EXPORT bool solveWithoutParticles(
			VolumeChannel & volChannel,
			VolumeChannel * outVolume,
			float d0,
			float d1,
			float tolerance = 1.0e-6f
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