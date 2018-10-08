#include "VolumeRasterization.h"
#include "VolumeRasterization.cuh"

#include "Volume.h"
#include "CudaUtility.h"

#include <iostream>

namespace blib {

	BLIB_EXPORT void blib::rasterize(
		float * meshTriangles, size_t triangleN,
		float * transformMatrices4x4, size_t instanceN,
		VolumeChannel & output)
	{

		CUDATimer t(true);
		Volume_Rasterize(meshTriangles, triangleN, transformMatrices4x4, instanceN, *output.getCUDAVolume());

		std::cout << "Rasterize " << instanceN << ": " << t.timeMs() << "ms" << std::endl;

	}

}

