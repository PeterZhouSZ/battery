#include "ConvexPolyhedron.h"

#include <fstream>
#include <set>
#include <algorithm>

namespace blib {

	



	/*BLIB_EXPORT void ConvexPolyhedron::recomputeNormals()
	{
		for (auto & f : faces) {
			Edge edgeA = { f.vertices[0], f.vertices[1] };
			Edge edgeB = { f.vertices[1], f.vertices[2] };
			vec3 edgeAvec = vertices[edgeA.x] - vertices[edgeA.y];
			vec3 edgeBvec = vertices[edgeB.x] - vertices[edgeB.y];
			f.normal = glm::normalize(glm::cross(edgeAvec, edgeBvec));
		}
	}*/
/*
	BLIB_EXPORT int ConvexPolyhedron::whichSide(const std::vector<vec3> & vertices, const vec3 & D, const vec3 & P) const
	{
		int positive = 0;
		int negative = 0;

		for (auto & vertex : vertices) {
			float t = glm::dot(D, vertex - P);
			if (t > 0.0f)
				positive++;
			else if (t < 0.0f)
				negative++;

			if (positive && negative)
				return 0;
		}

		return positive ? 1 : -1;
	}

	BLIB_EXPORT bool ConvexPolyhedron::intersects(const ConvexPolyhedron & other) const
	{
		for (auto & face : faces) {
			auto D = face.normal;

			if (whichSide(other.vertices, D, vertices[face.vertices.front()]) > 0)
				return false;

		}

		for (auto & otherFace : other.faces) {
			auto Dother = otherFace.normal;

			if (whichSide(vertices, Dother, other.vertices[otherFace.vertices.front()]) > 0)
				return false;
		}

		for (auto & edge : edges) {
			auto edgeVec = vertices[edge[0]] - vertices[edge[1]];
			for (auto & otherEdge : other.edges) {
				auto otherEdgeVec = other.vertices[otherEdge[0]] - other.vertices[otherEdge[1]];
				
				//auto D = glm::cross(edgeVec, otherEdgeVec);
				auto D = glm::normalize(glm::cross(edgeVec, otherEdgeVec));

				int side0 = whichSide(vertices, D, vertices[edge[0]]);
				if (side0 == 0)
					continue;

				int side1 = whichSide(other.vertices, D, vertices[edge[0]]);
				if (side1 == 0)
					continue;

				if (side0*side1 < 0)
					return false;

			}
		}

		return true;

	}
	*/
/*
	BLIB_EXPORT ConvexPolyhedron ConvexPolyhedron::transformed(const Transform & t) const
	{
		ConvexPolyhedron newcp = *this;

		for (auto & v : newcp.vertices) {
			v = t.transformPoint(v);
		}

		newcp.recomputeNormals();

		return newcp;
	}*/

	/*BLIB_EXPORT AABB ConvexPolyhedron::bounds() const
	{		
		AABB b;		
		for (auto & v : vertices) {
			b.min = glm::min(b.min, v);
			b.max = glm::max(b.max, v);
		}

		return b;		
	}*/

	/*BLIB_EXPORT ConvexPolyhedron ConvexPolyhedron::normalized(bool keepAspectRatio) const
	{
		auto b = bounds();

		Transform t;		
		vec3 range = b.max - b.min;
		if (keepAspectRatio) {
			float maxRange = glm::max(range.x, glm::max(range.y, range.z));
			range = vec3(maxRange);
		}
		t.scale = { 1.0f / range.x, 1.0f / range.y, 1.0f / range.z };
		
		t.translation = -(b.min * t.scale);
		
		return transformed(t);
	}*/

	/*BLIB_EXPORT std::vector<vec3> ConvexPolyhedron::flattenedTriangles() const
	{
		std::vector<vec3> result(faces.size() * 3);

		for (auto i = 0; i < faces.size(); i++) {
			auto & f = faces[i];
			assert(f.vertices.size() == 3);
			result[i * 3 + 0] = vertices[f.vertices[0]];
			result[i * 3 + 1] = vertices[f.vertices[1]];
			result[i * 3 + 2] = vertices[f.vertices[2]];		
		}

		return result;
	}
*/

}
