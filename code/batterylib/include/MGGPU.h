#pragma once

#include "BatteryLibDef.h"
#include "Types.h"
#include "Volume.h"

#include <array>

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
			Params params
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

		using SparseMat = int;
		using Vector = int;

		struct Level {			
			ivec3 dim;
			SparseMat A;
			SparseMat I;
			SparseMat R;
			Vector b;
			Vector x;
			Vector tmpx;
			Vector r;
			
		};

		std::vector<Level> _levels;


	};

	
}