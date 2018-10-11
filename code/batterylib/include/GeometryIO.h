#pragma once

#include "TriangleMesh.h"

#include <vector>
#include <fstream>

namespace blib {
	
	struct GeometryObject;
	
	//Custom format
	BLIB_EXPORT blib::TriangleMesh loadParticleMesh(const std::string & path);

	BLIB_EXPORT std::vector<std::shared_ptr<blib::GeometryObject>> readPosFile(std::ifstream & stream);

}