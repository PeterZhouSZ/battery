#pragma once


#include "Volume.h"
#include "BatteryLibDef.h"

namespace blib {

	/*
		Throws const char * exception on failure
		TODO: nothrow
	*/
	//BLIB_EXPORT Volume<unsigned char> loadTiffFolder(const char * folder);	
	BLIB_EXPORT VolumeChannel loadTiffFolder(const char * folder);

}