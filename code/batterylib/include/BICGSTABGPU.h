#pragma once

#include "BatteryLibDef.h"
#include "Volume.h"
#include "DataPtr.h"

#include <Eigen/Eigen>

#include "Solver.h"

struct CUDA_Volume;
struct DataPtr;

namespace blib {


	template <typename T>
	class BICGSTABGPU : public Solver<T> {

	public:
		BICGSTABGPU(bool verbose);



		bool prepare(const PrepareParams & params);
		Solver<T>::Output solve(const SolveParams & solveParams);

	private:

		//Linear system matrix
		std::shared_ptr<DataPtr> _A;
		std::shared_ptr<CUDA_Volume> _domain;
		std::shared_ptr<CUDA_Volume> _f;

		//Auxilliar buffer for reductions
		std::shared_ptr<DataPtr> _auxReduceBuffer;		
		

		//Vectors
		std::shared_ptr<CUDA_Volume> _temp;
		std::shared_ptr<CUDA_Volume> _x;
		std::shared_ptr<CUDA_Volume> _rhat0;
		std::shared_ptr<CUDA_Volume> _r, _p, _v, _h, _s, _t, _y, _z, _ainvert;


		//double _rho, _alpha, _omega, _beta;

				
	};

	

}
