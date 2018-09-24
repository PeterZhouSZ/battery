#include "VolumeMeasures.h"

#include "Volume.h"

#include "BICGSTABGPU.h"
#include "DiffusionSolver.h"
#include "MGGPU.h"

#include <iostream>

namespace blib {

	template <typename T>
	BLIB_EXPORT T getTortuosity(
		const VolumeChannel & mask, 
		const TortuosityParams & params, 
		DiffusionSolverType solverType, 
		VolumeChannel * concetrationOutput
	)
	{
		T tau = 0.0;
		auto maxDim = std::max(mask.dim().x, std::max(mask.dim().y, mask.dim().z));
		auto minDim = std::min(mask.dim().x, std::min(mask.dim().y, mask.dim().z));

		/*
			Either use provided output channel or create a temporary one
		*/
		VolumeChannel * outputChannel;
		std::unique_ptr<VolumeChannel> tmpChannel;
		if (concetrationOutput) {

			assert(concetrationOutput->type() == primitiveTypeof<T>());
			if (concetrationOutput->type() != primitiveTypeof<T>()) {
				std::cerr << "Invalid type of output volume channel" << std::endl;
				return T(0);
			}

			outputChannel = concetrationOutput;
		}
		else {
			if(solverType == DSOLVER_EIGEN)
				tmpChannel = std::make_unique<VolumeChannel>(mask.dim(), primitiveTypeof<T>());
			else
				tmpChannel = std::make_unique<VolumeChannel>(mask.dim(), TYPE_DOUBLE);
			outputChannel = tmpChannel.get();
		}
		
		
		/*
			Solver diffusion problem
		*/
		if (solverType == DSOLVER_BICGSTABGPU) {
			if (std::is_same<T, float>::value) {
				std::cerr << "Single precision not supported with BICGSTABGPU, using double instead." << std::endl;
				return T(0);
			}
			using K = double;
						
			BICGSTABGPU<K> solver(params.verbose);
			BICGSTABGPU<K>::PrepareParams p;
			p.dir = params.dir;
			p.d0 = params.coeffs[0];
			p.d1 = params.coeffs[1];			
			p.cellDim = vec3(1.0f / maxDim);
			p.mask = &mask;
			p.output = outputChannel;

			if (!solver.prepare(p))
				return T(0);

			BICGSTABGPU<K>::SolveParams sp;
			sp.maxIter = params.maxIter;
			sp.tolerance = params.tolerance;

			auto out = solver.solve(sp);	
			if (out.status != Solver<K>::SOLVER_STATUS_SUCCESS) {
				return T(0);
			}

			std::cout << "BICGSTAB ERROR: " << out.error << std::endl;
			
		}
		else if (solverType == DSOLVER_MGGPU) {
			if (std::is_same<T, float>::value) {
				std::cerr << "Single precision not supported with MGGPU, using double instead." << std::endl;				
			}
			using K = double;


			MGGPU<K> solver;

			MGGPU<K>::PrepareParams p;
			const int exactSolveDim = 4;
			p.dir = params.dir;
			p.d0 = params.coeffs[0];
			p.d1 = params.coeffs[1];
			p.cellDim = vec3(1.0f / maxDim);			
			p.levels = std::log2(minDim) - std::log2(exactSolveDim) + 1;


			if (!solver.prepare(mask, p, outputChannel))
				return T(0);

			MGGPU<K>::SolveParams sp;
			sp.alpha = 1.0;
			sp.verbose = params.verbose;
			sp.cycleType = MGGPU<double>::CycleType::V_CYCLE;
			sp.postN = 1; 
			sp.preN = 1;
			sp.tolerance = params.tolerance;

			K err = solver.solve(sp);		
			if (err > sp.tolerance) {
				return T(0);
			}

			std::cout << "MGGPU ERROR: " << err << std::endl;

		}
		else if (solverType == DSOLVER_EIGEN) {
			
			DiffusionSolver<T> solver(params.verbose);
			
			solver.prepare(
				mask, 
				params.dir,
				params.coeffs[0],
				params.coeffs[1]
			);
			
			double err = solver.solve(params.tolerance, params.maxIter, 256);				

			if (err > params.tolerance) {
				return T(0);
			}

			solver.resultToVolume(*outputChannel);
			outputChannel->getCurrentPtr().commit();
			
			std::cout << "EIGEN ERROR: " << err << std::endl;
		}
		
		/*
			Get porosity
		*/
		const T porosity = params.porosityPrecomputed ?
			static_cast<T>(params.porosity) :
			getPorosity<T>(mask);


		/*
			Tortuosity
		*/

		tau = porosity;


		



		return tau;
		
	}


	template <typename T>
	BLIB_EXPORT T getPorosity(const VolumeChannel & mask)
	{
		return T(mask.nonZeroElems()) / T(mask.totalElems());		
	}





	template BLIB_EXPORT double getTortuosity<double>(const VolumeChannel &, const TortuosityParams &, DiffusionSolverType, VolumeChannel *);
	template BLIB_EXPORT float getTortuosity<float>(const VolumeChannel &, const TortuosityParams &, DiffusionSolverType, VolumeChannel *);



	template BLIB_EXPORT float getPorosity<float>(const VolumeChannel &);
	template BLIB_EXPORT double getPorosity<double>(const VolumeChannel &);
}