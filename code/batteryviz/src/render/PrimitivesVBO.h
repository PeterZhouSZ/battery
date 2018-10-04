#pragma once

#include "VertexBuffer.h"

VertexBuffer<VertexData> getQuadVBO();

VertexBuffer<VertexData> getCubeVBO();

VertexBuffer<VertexData> getSphereVBO();


namespace blib {
	struct ConvexPolyhedron;
}

VertexBuffer<VertexData> getConvexPolyhedronVBO(const blib::ConvexPolyhedron & cp, vec4 color);