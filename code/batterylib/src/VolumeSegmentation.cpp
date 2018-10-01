#include "VolumeSegmentation.h"

#include "Volume.h"
#include "VolumeCCL.cuh"

namespace blib {


	const bool * VolumeCCL::getDirMask(Dir dir) const
	{
		return reinterpret_cast<const bool*>(boundaryLabelMask.data()) + (numLabels * uint(dir));
			
	}

	BLIB_EXPORT VolumeCCL getVolumeCCL(const VolumeChannel & mask, uchar background)
	{
		
		VolumeCCL ccl;

		ccl.labels = std::make_shared<VolumeChannel>(
			mask.dim(), TYPE_UINT, false, "CCLLabels"
		);		


		ccl.background = background;
		
		ccl.numLabels = VolumeCCL_Label(
			*mask.getCUDAVolume(), 
			*ccl.labels->getCUDAVolume(), 
			background
		);
		

		ccl.boundaryLabelMask.clear();
		ccl.boundaryLabelMask.resize(ccl.numLabels * 6);

		
		VolumeCCL_BoundaryLabels(
			*ccl.labels->getCUDAVolume(),
			ccl.numLabels, 
			(bool*)ccl.boundaryLabelMask.data()
		);


		/*std::array<std::vector<uint>, 6> boundaryLabels;
		int ptr = 0;
		for (auto i = 0; i < 6; i++) {
			for (auto j = 0; j < numLabels; j++) {
				if (boundaryMap[i * (numLabels)+j]) {
					boundaryLabels[i].push_back(j);
				}
			}
			std::cout << "Boundary: " << i << " has " << boundaryLabels[i].size() << " labels" << std::endl;
		}*/

		

		return ccl;
	}


	BLIB_EXPORT VolumeChannel generateBoundaryConnectedVolume(const VolumeCCL & ccl, Dir dir)
	{
		VolumeChannel boundary(ccl.labels->dim(), TYPE_UCHAR, false, "CCLBoundary");

		VolumeCCL_GenerateVolume(
			*ccl.labels->getCUDAVolume(),
			ccl.numLabels,
			ccl.getDirMask(dir),
			*boundary.getCUDAVolume()
		);

		return boundary;
	}

	BLIB_EXPORT VolumeChannel generateCCLVisualization(
		const VolumeCCL & ccl,
		VolumeChannel * mask
	)
	{
		VolumeChannel outViz(ccl.labels->dim(), TYPE_UCHAR4, false, "CCLViz");
		if (mask) {
			VolumeCCL_Colorize(*ccl.labels->getCUDAVolume(), *outViz.getCUDAVolume(), mask->getCUDAVolume().get(), 
				(ccl.background) ? 0 : 255
				);
		}
		else {
			VolumeCCL_Colorize(*ccl.labels->getCUDAVolume(), *outViz.getCUDAVolume());
		}
		return outViz;		
	}

	

}

