#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#include "DataPtr.h"

#include <array>
#include <Eigen/Eigen>

//Forward declare cuda type holding the volume
struct CUDA_Volume; //TODO move to blib namespace


namespace blib {

	class Volume;
	class VolumeChannel;


	template <typename T>

	class MGGPU {
	public: 

		enum CycleType {
			V_CYCLE,
			W_CYCLE,
			V_CYCLE_SINGLE
		};

		struct PrepareParams {
			Dir dir;
			T d0;
			T d1;
			uint levels;
			vec3 cellDim;
		};

		struct SolveParams {
			T alpha = 1.0;
			int preN = 1;
			int postN = 1;
			CycleType cycleType = V_CYCLE;
			T tolerance = 1e-6;
			size_t maxIter = 1024;
			bool verbose = false;
			bool verboseDebug = false;
		};

		using Vector = Eigen::VectorXd;
		using SparseMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
		using DenseMat = Eigen::MatrixXd;
		using DirectSolver = Eigen::SparseLU<SparseMat>;


		BLIB_EXPORT MGGPU(bool verbose = false);
		
		BLIB_EXPORT bool prepare(
			const VolumeChannel & mask,
			PrepareParams params,	
			VolumeChannel * output
		);



		//Returns error
		BLIB_EXPORT T solve(
			const SolveParams & solveParams			
		);
		
		BLIB_EXPORT void reset();

		BLIB_EXPORT size_t iterations() const {
			return _iterations;
		}

		BLIB_EXPORT T porosity() const {
			return _porosity;
		}


	private:


		bool alloc(int levelLimit = -1);
		
		int numLevels() const { return _params.levels; }

		std::vector<int> genCycle(CycleType ctype) const;


		bool saveVolume(CUDA_Volume & v, const std::string & name, int level);
		void commitVolume(CUDA_Volume & v);
		void retrieveVolume(CUDA_Volume & v);
		

		PrepareParams _params;
		const VolumeChannel * _mask;
		std::unique_ptr<Volume> _volume;

		//using SparseMat = int; //Todo cusparse mat
		//using Vector = int; //Todo cusparse vec

		struct Level {			
			ivec3 dim;
			 
			std::shared_ptr<CUDA_Volume> domain;
			std::shared_ptr<CUDA_Volume> x;
			std::shared_ptr<CUDA_Volume> tmpx;
			std::shared_ptr<CUDA_Volume> r;
			std::shared_ptr<CUDA_Volume> f;

			DataPtr A;
			DataPtr I;

			size_t N() const { return dim.x*dim.y*dim.z; }
			
#ifdef MGGPU_CPU_TEST
			SparseMat Acpu;
#endif
			
		};



		std::vector<Level> _levels;
		
		SparseMat _lastLevelA;
		DirectSolver _lastLevelSolver;

		DataPtr _auxReduceBuffer;

		size_t _iterations;

		T _porosity;

		VolumeChannel * _output;


		bool _verbose;

	};

	
	
	
	
	
	
	
}