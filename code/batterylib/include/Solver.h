#pragma once
#include "Volume.h"

namespace blib {

	template <typename T>
	class Solver {


	public:

		Solver(bool verbose) : _verbose(verbose), _iterations(0) {

		}

		struct PrepareParams {
			Volume * volume = nullptr;
			int maskID = 0;
			Dir dir = X_NEG;
			T d0 = 0.0;
			T d1 = 1.0;
			vec3 cellDim = vec3(1.0,1.0,1.0);					
		};

		struct SolveParams {
			T tolerance;
			size_t maxIter;			
		};

		enum Status {
			SOLVER_STATUS_SUCCESS,
			SOLVER_STATUS_DIVERGED,
			SOLVER_STATUS_NO_CONVERGENCE
		};

		struct Output {
			T error;
			Status status;
			size_t iterations;
		};

		virtual bool prepare(const PrepareParams & params) = 0;
		virtual Output solve(const SolveParams & solveParams) = 0;

	protected:
		bool _verbose;
		PrepareParams _params;
		size_t _iterations;
	};


}