#pragma once

#include "BatteryLibDef.h"
#include "Volume.h"

#include <cusparse.h>
#include <cusolverSp.h>


namespace blib {

	class DiffusionSolver {


	public:
		BLIB_EXPORT DiffusionSolver();
		BLIB_EXPORT ~DiffusionSolver();


		BLIB_EXPORT bool prepare(VolumeChannel & volChannel, int d);

	private:
			


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