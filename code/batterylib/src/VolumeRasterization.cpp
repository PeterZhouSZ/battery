#include "VolumeRasterization.h"
#include "VolumeRasterization.cuh"

#include "Volume.h"
#include "CudaUtility.h"

#include "GeometryObject.h"
#include "TriangleMesh.h"

#include <iostream>

namespace blib {

	BLIB_EXPORT void blib::rasterize(
		const float * meshTriangles, size_t triangleN,
		const float * transformMatrices4x4, size_t instanceN,
		VolumeChannel & output)
	{

		CUDATimer t(true);
		Volume_Rasterize(meshTriangles, triangleN, transformMatrices4x4, instanceN, *output.getCUDAVolume());

		std::cout << "Rasterize " << instanceN << ": " << t.timeMs() << "ms" << std::endl;

	}

	BLIB_EXPORT void rasterize(
		const std::vector<std::shared_ptr<GeometryObject>> & objects, 
		VolumeChannel & output
	)
	{
		
		std::unordered_map<
			std::shared_ptr<Geometry>,
			std::vector<mat4>
		> templateToInstance;


		for (const auto & obj : objects) {
			const auto & tmpGeom = obj->getTemplateGeometry();

			templateToInstance[tmpGeom].push_back(obj->getTransform().getAffine());
		}

		for (auto it : templateToInstance) {
			
			auto tm = std::dynamic_pointer_cast<TriangleMesh>(it.first);

			//Not a 
			if (!tm) {
				assert("Unsupported rasterization");
				continue;
			}		

			rasterize(
				reinterpret_cast<float*>(tm->getTriangleArray().data()),
				tm->faces.size(),
				reinterpret_cast<float*>(it.second.data()),
				it.second.size(),
				output
			);		
		}



	}

}

