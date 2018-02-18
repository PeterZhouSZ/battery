#pragma once

#include "BatteryLibDef.h"
#include "Volume.h"


#include <Eigen/Eigen>
#include <Eigen/IterativeLinearSolvers>

namespace blib {

	template <typename T>
	class DiffusionSolver {


	public:

		using value_type = T;

		BLIB_EXPORT DiffusionSolver(bool verbose = true);
		BLIB_EXPORT ~DiffusionSolver();


		/*
			Assumes volChannel is synchronized on cpu
		*/
		BLIB_EXPORT bool prepare(
			VolumeChannel & volChannel,
			Dir dir,
			T d0,
			T d1
		);

		//Returns current error
		BLIB_EXPORT T solve(
			T tolerance,
			size_t maxIterations,
			size_t iterPerStep = size_t(-1)
		);

		//Returns false there's format mismatch and no conversion available
		BLIB_EXPORT bool resultToVolume(VolumeChannel & vol);

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

		BLIB_EXPORT T tortuosity(
			const VolumeChannel & mask,			
			Dir dir
		);

		BLIB_EXPORT T porosity() const { return _porosity; }

	private:
			
		
		bool _verbose;

		Eigen::Matrix<T, Eigen::Dynamic, 1> _rhs;
		Eigen::Matrix<T, Eigen::Dynamic, 1> _x;
		Eigen::SparseMatrix<T, Eigen::RowMajor> _A;

		Eigen::BiCGSTAB<Eigen::SparseMatrix<T, Eigen::RowMajor>> _solver;


		T _porosity;

	};


	
}