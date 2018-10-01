#pragma once

#include "BatteryLibDef.h"
#include "Types.h"


#include <memory>
#include <vector>


namespace blib {
	
	class Volume;
	class VolumeChannel;


	struct VolumeCCL {
		uint numLabels; //Includes 0 label -> background		
		std::shared_ptr<VolumeChannel> labels;		
		const bool * getDirMask(Dir dir) const;

		std::vector<uchar> boundaryLabelMask;
		uchar background;

	};

	BLIB_EXPORT VolumeCCL getVolumeCCL(
		const VolumeChannel & mask,
		uchar background
	);

	BLIB_EXPORT VolumeChannel generateBoundaryConnectedVolume(
		const VolumeCCL & ccl,
		Dir dir
	);

	BLIB_EXPORT VolumeChannel generateCCLVisualization(
		const VolumeCCL & ccl,
		VolumeChannel * mask = nullptr
	);


}