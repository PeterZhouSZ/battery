#pragma once

#include "VertexBuffer.h"

VertexBuffer<VertexData> getQuadVBO();

VertexBuffer<VertexData> getCubeVBO();

VertexBuffer<VertexData> getSphereVBO();


namespace blib {
	struct ConvexPolyhedron;
	struct TriangleMesh;
}

VertexBuffer<VertexData> getTriangleMeshVBO(const blib::TriangleMesh & cp, vec4 color);