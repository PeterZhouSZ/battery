#pragma once


#include "BatteryLibDef.h"
#include "Types.h"

#include <vector>


namespace blib {


	using ConvexPolyhedronHalfplane = vec4;

	struct ConvexPolyhedron{
		
		using Edge = ivec2;
		struct Face {			
			std::vector<int> vertices;
			vec3 normal;
		};

		std::vector<vec3> vertices;
		std::vector<Face> faces;
		std::vector<Edge> edges; 
			

		BLIB_EXPORT static ConvexPolyhedron loadFromFile(const std::string & path);

		BLIB_EXPORT int whichSide(const std::vector<vec3> & vertices, const vec3 & D, const vec3 & P) const;

		BLIB_EXPORT bool intersects(const ConvexPolyhedron & other) const;

	};

}