#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#include <Eigen/Eigen>
#include <Eigen/IterativeLinearSolvers>

#include <array>

namespace blib {
	template <typename T>
	class MultigridSolver {

		
		using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
		using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

	public:

		BLIB_EXPORT MultigridSolver(bool verbose);

		BLIB_EXPORT bool prepare(const uchar * D, ivec3 dim, Dir dir, T d0, T d1, uint levels);

		BLIB_EXPORT T solve(
			T tolerance,
			size_t maxIterations			
		);

	private:

		BLIB_EXPORT bool prepareAtLevel(
			const float * D, ivec3 dim, Dir dir, 
			uint level
		);

		std::vector<SparseMat> _A;		
		std::vector<Vector> _rhs;
		std::vector<Vector> _x;

		bool _verbose;
		uint _lv; //0 = fine level, _lv-1 = coarsest level

		std::array<T,27> _restrictOp;

	};

}