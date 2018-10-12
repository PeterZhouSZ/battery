#pragma once

#include "TriangleMesh.h"

#include <vector>
#include <fstream>

namespace blib {
	
	struct GeometryObject;
	
	//Custom format
	BLIB_EXPORT blib::TriangleMesh loadParticleMesh(const std::string & path);

	BLIB_EXPORT std::vector<std::shared_ptr<blib::GeometryObject>> readPosFile(
		std::ifstream & stream, size_t index = 0, AABB trim = AABB::unit());

	BLIB_EXPORT size_t getPosFileCount(std::ifstream & stream);

}