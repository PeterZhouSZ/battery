#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#include "Volume.h"
#include <vector>

#include <cusparse.h>
#include <cusolverSp.h>

namespace blib {

	template <typename T>
	class MultigridGPU {

	public:

		enum CycleType {
			V_CYCLE,
			W_CYCLE
		};

		BLIB_EXPORT MultigridGPU(bool verbose);

		BLIB_EXPORT bool prepare(			
			const VolumeChannel & mask,
			const VolumeChannel & concetration,
			Dir dir,
			T d0, T d1,
			uint levels,
			vec3 cellDim
		);

		BLIB_EXPORT T solve(			
			T tolerance,
			size_t maxIterations,
			CycleType cycleType = W_CYCLE
		);

		BLIB_EXPORT void setDebugVolume(Volume * vol) {
			_debugVolume = vol;
		}

		BLIB_EXPORT void setVerbose(bool val) {
			_verbose = val;
		}

	private:

		void prepareSystemAtLevel(uint level);

		static std::vector<int> genCycle(CycleType ctype, uint levels);

		T squareNorm(Texture3DPtr & surf, ivec3 dim);

		std::vector<DataPtr> _A;
		std::vector<Texture3DPtr> _D;
		std::vector<Texture3DPtr> _f;
		std::vector<Texture3DPtr> _x;
		std::vector<Texture3DPtr> _tmpx;
		std::vector<Texture3DPtr> _r;


		//Last level direct solve
		DataPtr _xLast;
		DataPtr _fLast;
		DataPtr _ALast;
		DataPtr _ALastRowPtr;
		DataPtr _ALastColInd;
		size_t _LastNNZ;

		cusolverSpHandle_t _cusolverHandle = nullptr;
		cusparseHandle_t _cusparseHandle = nullptr;
		cudaStream_t _stream = nullptr;
		cusparseMatDescr_t _descrA = nullptr;



		DataPtr _auxReduceBuffer;

		Volume * _debugVolume;
		vec3 _cellDim;
		bool _verbose;
		uint _lv; //0 = fine level, _lv-1 = coarsest level
		std::vector<ivec3> _dims;
		uint _iterations;
		Dir _dir;

		const PrimitiveType _type;

		std::vector<cudaStream_t> _streams;

	};

	

	

}