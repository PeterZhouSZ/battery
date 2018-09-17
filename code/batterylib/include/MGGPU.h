#pragma once

#include "BatteryLibDef.h"
#include "Types.h"
#include "Volume.h"

#include <array>

#include "MGGPU_Types.h"

#include <Eigen/Eigen>

//#define MGGPU_CPU_TEST

namespace blib {


	template <typename T>

	class MGGPU {
	public: 

		enum CycleType {
			V_CYCLE,
			W_CYCLE,
			V_CYCLE_SINGLE
		};

		using Vector = Eigen::VectorXd;
		using SparseMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
		using DenseMat = Eigen::MatrixXd;
		using DirectSolver = Eigen::SparseLU<SparseMat>;


		BLIB_EXPORT MGGPU();

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
			
		};



		BLIB_EXPORT bool prepare(
			VolumeChannel & mask,
			PrepareParams params,
			Volume & volume
		);



		//Returns error
		BLIB_EXPORT T solve(
			const SolveParams & solveParams
			/*T tolerance, 
			size_t maxIter, 
			CycleType cycleType = W_CYCLE,
			T alpha = 1.0 //Smoothing over/under relaxation*/
		);

		BLIB_EXPORT T tortuosity();
		

		BLIB_EXPORT void profile();

		BLIB_EXPORT void reset();

		BLIB_EXPORT size_t iterations() const {
			return _iterations;
		}

		BLIB_EXPORT T porosity() const {
			return _porosity;
		}


	private:


		bool alloc();
		
		int numLevels() const { return _params.levels; }

		std::vector<int> genCycle(CycleType ctype) const;


		bool saveVolume(MGGPU_Volume & v, const std::string & name, int level);
		void commitVolume(MGGPU_Volume & v);
		void retrieveVolume(MGGPU_Volume & v);
		

		PrepareParams _params;
		VolumeChannel * _mask;
		Volume * _volume;

		//using SparseMat = int; //Todo cusparse mat
		//using Vector = int; //Todo cusparse vec

		struct Level {			
			ivec3 dim;
			MGGPU_Volume domain;

			MGGPU_Volume x;
			MGGPU_Volume tmpx;
			MGGPU_Volume r;
			MGGPU_Volume f;

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
	};

	
	
	
	
	
}