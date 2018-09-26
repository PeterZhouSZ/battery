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

		//Porosity
		bool porosityPrecomputed = false;
		double porosity = -1.0;
	};

	/*
		Calculates tortuosity
		Inputs: 
			binary volume mask
			Tortuosity params detailing direction, tolerance, etc.
			Solver to use,
		Outputs:
			Returns tortuosity.
			Returns 0 on error
			If concetrationOutput is specified, 
			the concetration from diffusion equation is stored there.
		Notes:
			Calculates porosity if not provided

	*/
	template <typename T> 
	BLIB_EXPORT T getTortuosity(
		const VolumeChannel & mask,  
		const TortuosityParams & params,
		DiffusionSolverType solverType = DSOLVER_BICGSTABGPU,
		VolumeChannel * concetrationOutput = nullptr //Optional output of diffusion
	);

	template <typename T>
	BLIB_EXPORT T getPorosity(const VolumeChannel & mask);


	template <typename T>
	BLIB_EXPORT T getReactiveAreaDensity(
		const VolumeChannel & mask, 
		ivec3 res, float isovalue, float smooth,
		uint * vboOut = nullptr, //If not null, a triangle mesh will be generated and saved to vbo
		size_t * NvertsOut = nullptr
	);
	
	}