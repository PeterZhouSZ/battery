#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

namespace blib {

	class Volume;
	class VolumeChannel;
	

	enum DiffusionSolverType {
		DSOLVER_BICGSTABGPU,
		DSOLVER_MGGPU,
		DSOLVER_EIGEN
	};

	struct TortuosityParams {
		Dir dir = X_NEG;
		vec2 coeffs = vec2(1.0, 0.001);
		double tolerance = 1e-6;
		size_t maxIter = 10000;
		bool verbose = false;
	};

	template <typename T> 
	BLIB_EXPORT T tortuosity(
		const VolumeChannel & mask,  
		const TortuosityParams & params,
		DiffusionSolverType solverType = DSOLVER_BICGSTABGPU,
		VolumeChannel * concetrationOutput = nullptr //Optional output of diffusion
	);

	template <typename T>
	BLIB_EXPORT T porosity(Volume & volume, int maskID);


	template <typename T>
	BLIB_EXPORT T reactiveAreaDensity(Volume & volume, int maskID);
	
}