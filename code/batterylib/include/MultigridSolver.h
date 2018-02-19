#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#include <Eigen/Eigen>
#include <Eigen/IterativeLinearSolvers>

#include "Volume.h"

#include <array>

namespace blib {
	template <typename T>
	class MultigridSolver {

		
		using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
		using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

	public:

		BLIB_EXPORT MultigridSolver(bool verbose);

		BLIB_EXPORT bool prepare(
			Volume & v, //to remove, for debug only
			const uchar * D, 
			ivec3 dim, 
			Dir dir, 
			T d0, T d1, 
			uint levels);

		BLIB_EXPORT T solve(
			Volume &v,  //to remove, for debug only
			T tolerance,
			size_t maxIterations			
		);

		BLIB_EXPORT bool resultToVolume(VolumeChannel & vol);

		BLIB_EXPORT bool generateErrorVolume(Volume & vol);

		BLIB_EXPORT T tortuosity(
			const VolumeChannel & mask,
			Dir dir
		);

		BLIB_EXPORT void setVerbose(bool val) {
			_verbose = val;
		}

	private:

		BLIB_EXPORT bool prepareAtLevel(
			const float * D, ivec3 dim, Dir dir, 
			uint level
		);

		std::vector<SparseMat> _A;		
		std::vector<Vector> _f;
		std::vector<Vector> _x;
		std::vector<Vector> _tmpx;
		std::vector<Vector> _r;

		bool _verbose;
		uint _lv; //0 = fine level, _lv-1 = coarsest level
		std::vector<ivec3> _dims;

		std::array<T, 27> _restrictOp;
		std::array<T, 27> _interpOp;

		T _porosity;

	};

}