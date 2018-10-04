#pragma once


#include "BatteryLibDef.h"
#include "Types.h"

#include <vector>


namespace blib {


	using ConvexPolyhedronHalfplane = vec4;

	struct ConvexPolyhedron{
		
		using Edge = ivec2;
		struct Face {
			std::vector<int> edges;
			std::vector<int> vertices;
			vec3 normal;
		};

		std::vector<vec3> vertices;
		std::vector<Face> faces;
		std::vector<Edge> edges; //ordering same as faces?


		ConvexPolyhedron(std::vector<vec3> & A, std::vector<float> & b) {

		
		}

		int whichSide(const std::vector<vec3> & vertices, const vec3 & D, const vec3 & P) const{
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

		bool intersects(const ConvexPolyhedron & other) const {


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
					auto D = glm::cross(edgeVec, otherEdgeVec);

					int side0 = whichSide(vertices, D, vertices[edge[0]]);
					if (side0 == 0)
						continue;

					int side1 = whichSide(other.vertices, D, other.vertices[otherEdge[0]]);
					if (side1 == 0)
						continue;

					if (side0*side1 < 0)
						return false;

				}

			
			}


		}

	};

}