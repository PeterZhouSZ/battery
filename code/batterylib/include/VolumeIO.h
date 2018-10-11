#pragma once


#include "Volume.h"
#include "BatteryLibDef.h"

namespace blib {

	/*
		Throws const char * exception on failure
		TODO: nothrow
	*/
	//BLIB_EXPORT Volume<unsigned char> loadTiffFolder(const char * folder);	
	BLIB_EXPORT VolumeChannel loadTiffFolder(const char * folder, bool commitToGPU = true);

	BLIB_EXPORT bool saveVolumeBinary(const char * path, const VolumeChannel & channel);
	BLIB_EXPORT VolumeChannel loadVolumeBinary(const char * path);





}