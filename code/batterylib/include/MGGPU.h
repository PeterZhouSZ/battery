#pragma once

#include "BatteryLibDef.h"
#include "Types.h"
#include "Volume.h"

#include <array>

#include "../src/cuda/MGGPU.cuh"

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

		struct Params {
			Dir dir;
			T d0;
			T d1;
			uint levels;
			vec3 cellDim;
		};

		BLIB_EXPORT bool prepare(
			const VolumeChannel & mask,
			Params params,
			Volume & volume
		);



		//Returns error
		BLIB_EXPORT T solve(T tolerance, size_t maxIter, CycleType cycleType = W_CYCLE);

		BLIB_EXPORT T tortuosity();

		BLIB_EXPORT void profile();



	private:


		bool alloc();
		
		int numLevels() const { return _params.levels; }

		std::vector<int> genCycle(CycleType ctype) const;


		bool saveVolume(MGGPU_Volume & v, const std::string & name, int level);
		void commitVolume(MGGPU_Volume & v);
		void retrieveVolume(MGGPU_Volume & v);
		

		Params _params;
		const VolumeChannel * _mask;
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

	};

	
	
	
}