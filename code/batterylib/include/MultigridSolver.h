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

		
		

	public:

		using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
		using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

		BLIB_EXPORT MultigridSolver(bool verbose);

		BLIB_EXPORT bool prepare(
			Volume & v, //to remove, for debug only
			const uchar * D, 
			ivec3 dim, 
			Dir dir, 
			T d0, T d1, 
			uint levels,
			vec3 cellDim
		);

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

		BLIB_EXPORT uint iterations() { 
			return _iterations; 
		}

		BLIB_EXPORT const Vector & result() const {
			return _x[0];
		}

	private:


		BLIB_EXPORT bool prepareAtLevelFVM(
			const T * D, ivec3 dim, Dir dir,
			uint level
		);

		std::vector<SparseMat> _A;		
		std::vector<SparseMat> _I;
		std::vector<SparseMat> _R;
		std::vector<Vector> _f;
		std::vector<Vector> _x;
		std::vector<Vector> _tmpx;
		std::vector<Vector> _r;
		glm::tvec3<T, glm::highp> _cellDimFine;

		bool _verbose;
		uint _lv; //0 = fine level, _lv-1 = coarsest level
		std::vector<ivec3> _dims;

		std::array<T, 27> _restrictOp;
		std::array<T, 27> _interpOp;

		T _porosity;
		uint _iterations;

		std::vector<std::vector<T>> _D;

		Dir _dir;

	};

}