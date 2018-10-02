#include "VolumeMeasures.h"

#include "Volume.h"

#include "BICGSTABGPU.h"
#include "DiffusionSolver.h"
#include "MGGPU.h"
#include "VolumeSegmentation.h"

#include "VolumeSurface.cuh"
#include "VolumeSurface.h"


#include "../src/cuda/CudaUtility.h"

#include <numeric>
#include <iostream>
#include <vector>

#include <glm/gtc/constants.hpp>

namespace blib {

	template <typename T>
	BLIB_EXPORT T getTortuosity(
		const VolumeChannel & mask, 
		const TortuosityParams & params, 
		DiffusionSolverType solverType, 
		VolumeChannel * concetrationOutput
	)
	{
		if (mask.type() != TYPE_UCHAR) {
			std::cerr << "Mask must by type uchar" << std::endl;
			return T(0);
		}

		if (mask.getCurrentPtr().getCPU() == nullptr) {
			std::cerr << "Mask must be allocated and retrieved to CPU" << std::endl;
			return T(0);
		}	


		
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



		auto & concPtr = outputChannel->getCurrentPtr();		
		
		//Make sure is allocated on CPU
		concPtr.allocCPU();	

		//Retrieve whole volume (TODO: retrieve just slice)
		concPtr.retrieve();	

		const auto & maskPtr = mask.getCurrentPtr();

		{
			T * concetration = static_cast<T*>(concPtr.getCPU());
			const uchar * maskData = static_cast<const uchar*>(maskPtr.getCPU());
			const ivec3 dim = mask.dim();

			const int primaryDim = getDirIndex(params.dir);
			const int secondaryDims[2] = { (primaryDim + 1) % 3, (primaryDim + 2) % 3 };
			const T cellDim = T(1) / static_cast<T>(maxDim);

			//Number of elems in plane
			const int n = dim[secondaryDims[0]] * dim[secondaryDims[1]];

			//Index in primary dim
			const int k = (getDirSgn(params.dir) == -1) ? 0 : dim[primaryDim] - 1;
			
			std::vector<T> isums(dim[secondaryDims[0]], T(0));

			#pragma omp parallel for
			for (auto i = 0; i < dim[secondaryDims[0]]; i++) {
				T jsum = T(0);
				for (auto j = 0; j < dim[secondaryDims[1]]; j++) {
					ivec3 pos;
					pos[primaryDim] = k;
					pos[secondaryDims[0]] = i;
					pos[secondaryDims[1]] = j;

					if (maskData[linearIndex(dim, pos)] == 0)
						jsum += concetration[linearIndex(dim, pos)];
				}
				isums[i] = jsum;
			}

			T sum = std::accumulate(isums.begin(), isums.end(), T(0));			
			T dc = sum / n;			

			tau = porosity / (dc * dim[primaryDim] * 2);		
		}



		return tau;
		
	}


	template <typename T>
	BLIB_EXPORT T getPorosity(const VolumeChannel & mask)
	{
		return T(mask.sumZeroElems()) / T(mask.totalElems());		
	}

	template <typename T>
	BLIB_EXPORT T getReactiveAreaDensity(const VolumeChannel & mask, ivec3 res, float isovalue)
	{
		VolumeChannel areas = getVolumeArea(mask, res, isovalue);
		T area = 0.0f;
		areas.sum(&area);		
		return area;		
	}

	template <typename T>
	BLIB_EXPORT std::array<T, 6> getReactiveAreaDensityTensor(const VolumeCCL & ccl, ivec3 res, float isovalue)
	{

		std::array<T, 6> result;

		VolumeSurface_MCParams params;
		if (res.x == 0 || res.y == 0 || res.z == 0) {
			params.res = make_uint3(ccl.labels->dim().x, ccl.labels->dim().y, ccl.labels->dim().z);
		}
		else {
			params.res = make_uint3(res.x, res.y, res.z);
		}
		params.isovalue = isovalue;
		params.smoothingOffset = 1.0f;

		
		for (auto i = 0; i < 6; i++) {
			
			//Generate volume of empty space that is connected to i'th boundary
			VolumeChannel boundaryVolume = blib::generateBoundaryConnectedVolume(ccl, Dir(i));
			boundaryVolume.getCurrentPtr().createTexture();

			//Run marching cubes on the volume
			VolumeChannel areas = getVolumeArea(boundaryVolume, res, isovalue);

			T area = 0.0f;
			areas.sum(&area);
			result[i] = area;
		}

		return result;
	}

	template <typename T>
	BLIB_EXPORT T getShapeFactor(T averageParticleArea, T averageParticleVolume)
	{	
		const T numeratorCoeff = glm::pow(T(3.0 / (4.0 * glm::pi<T>())), T(1.0 / 3.0));

		return (numeratorCoeff * averageParticleArea) / glm::pow(averageParticleVolume, T(2.0 / 3.0));
	}

	
	

	template BLIB_EXPORT double getTortuosity<double>(const VolumeChannel &, const TortuosityParams &, DiffusionSolverType, VolumeChannel *);
	template BLIB_EXPORT float getTortuosity<float>(const VolumeChannel &, const TortuosityParams &, DiffusionSolverType, VolumeChannel *);



	template BLIB_EXPORT float getPorosity<float>(const VolumeChannel &);
	template BLIB_EXPORT double getPorosity<double>(const VolumeChannel &);

	template BLIB_EXPORT float getReactiveAreaDensity<float>(const VolumeChannel &, ivec3, float);
	template BLIB_EXPORT double getReactiveAreaDensity<double>(const VolumeChannel &, ivec3, float);

	template BLIB_EXPORT std::array<double, 6> getReactiveAreaDensityTensor(const VolumeCCL & ccl, ivec3 res, float isovalue);
	template BLIB_EXPORT std::array<float, 6> getReactiveAreaDensityTensor(const VolumeCCL & ccl, ivec3 res, float isovalue);


	template BLIB_EXPORT double getShapeFactor<double>(double,double);
	template BLIB_EXPORT float getShapeFactor<float>(float, float);
}
