#pragma once

#include "BatteryLibDef.h"
#include "Types.h"
#include "Volume.h"

#include <array>

#include "../src/cuda/MGGPU.cuh"

namespace blib {


	template <typename T>

	class MGGPU {
	public: 

		enum CycleType {
			V_CYCLE,
			W_CYCLE,
			V_CYCLE_SINGLE
		};

		BLIB_EXPORT MGGPU();

		struct Params {
			Dir dir;
			T d0;
			T d1;
			uint levels;
			vec3 cellDim;
		};

		BLIB_EXPORT bool prepare(
			const VolumeChannel & mask,
			Params params,
			Volume & volume
		);




		BLIB_EXPORT void solve(T tolerance, size_t maxIter, CycleType cycleType = W_CYCLE);

		BLIB_EXPORT T tortuosity();



	private:


		bool alloc();
		bool buildLinearSystem();
		bool subsampleDiffusion();

		bool buildLevelsGalerkin();

		int numLevels() const { return _params.levels; }


		Params _params;
		const VolumeChannel * _mask;
		Volume * _volume;

		using SparseMat = int; //Todo cusparse mat
		using Vector = int; //Todo cusparse vec

		struct Level {			
			ivec3 dim;
			MGGPU_Volume domain;

			MGGPU_Volume x;
			MGGPU_Volume tmpx;
			MGGPU_Volume r;
			MGGPU_Volume f;

			DataPtr A;

			size_t N() const { return dim.x*dim.y*dim.z; }
			/*SparseMat A;
			SparseMat I;
			SparseMat R;
			Vector b;
			Vector x;
			Vector tmpx;
			Vector r;*/
			
		};

		std::vector<Level> _levels;


	};

	
}