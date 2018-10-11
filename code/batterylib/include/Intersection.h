#pragma once


#include "BatteryLibDef.h"
/*
	bool isectTestXY(); .. only tests
	IsectXYResult isectXY(); .. returns struct with details
*/

namespace blib {

	struct TriangleMesh;
	struct Geometry;

	BLIB_EXPORT bool isectTestConvexMesh(const TriangleMesh & a, const TriangleMesh & b);


	BLIB_EXPORT bool isectTest(const Geometry & a, const Geometry & b);
	



}
