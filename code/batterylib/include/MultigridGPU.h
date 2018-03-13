#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#include "Volume.h"
#include <vector>

namespace blib {

	template <typename T>
	class MultigridGPU {

	public:
		BLIB_EXPORT MultigridGPU(bool verbose);

		BLIB_EXPORT bool prepare(
			Volume & volume,
			const VolumeChannel & mask,
			const VolumeChannel & concetration,
			Dir dir,
			T d0, T d1,
			uint levels,
			vec3 cellDim
		);

		BLIB_EXPORT T solve(			
			T tolerance,
			size_t maxIterations
		);

	private:
		std::vector<DataPtr> _A;
		std::vector<Texture3DPtr> _D;
		std::vector<Texture3DPtr> _f;
		std::vector<Texture3DPtr> _x;
		//std::vector<Texture3DPtr> _tmpx;
		std::vector<Texture3DPtr> _r;

		vec3 _cellDim;
		bool _verbose;
		uint _lv; //0 = fine level, _lv-1 = coarsest level
		std::vector<ivec3> _dims;
		uint _iterations;

		const PrimitiveType _type;

	};


	

}