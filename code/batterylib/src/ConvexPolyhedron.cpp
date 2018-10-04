#include "ConvexPolyhedron.h"

#include <fstream>
#include <set>
#include <algorithm>

namespace blib {

	BLIB_EXPORT blib::ConvexPolyhedron blib::ConvexPolyhedron::loadFromFile(const std::string & path)
	{

		std::ifstream f(path);

		if (!f.good())
			throw "ConvexPolyhedron::loadFromFile invalid file";

		ConvexPolyhedron cp;
		

		int nv, nf;
		f >> nv;

		if(nv <= 0)
			throw "ConvexPolyhedron::loadFromFile invalid number of vertices";

		cp.vertices.resize(nv);

		for (auto i = 0; i < nv; i++) {
			f >> cp.vertices[i].x >> cp.vertices[i].y >> cp.vertices[i].z;
		}

		f >> nf;

		if (nf <= 0)
			throw "ConvexPolyhedron::loadFromFile invalid number of faces";

		cp.faces.resize(nf);

		auto cmpEdge = [](const Edge & a, const Edge & b) {
			Edge as = a;
			if (as.x > as.y) std::swap(as.x, as.y);
			Edge bs = b;
			if (bs.x > bs.y) std::swap(bs.x, bs.y);

			if (as.x < bs.x)
				return true;
			if (as.x > bs.x)
				return false;
			return (as.y < bs.y);
		};

		std::set<Edge, decltype(cmpEdge)> edgeSet(cmpEdge);

		
		for (auto i = 0; i < nf; i++) {
			
			Face & face = cp.faces[i];	

			//Read number of vertices in this face
			int nfv;
			f >> nfv;

			assert(nfv >= 3);

			//Load vertex indices
			face.vertices.resize(nfv);
			for (auto k = 0; k < nfv; k++) {
				f >> face.vertices[k];
			}

			//Save edges
			for (auto k = 0; k < nfv; k++) {
				Edge e = { face.vertices[k], face.vertices[(k + 1) % nfv] };
				edgeSet.insert(e);				
			}

			Edge edgeA = { face.vertices[0], face.vertices[1] };			
			Edge edgeB = { face.vertices[2], face.vertices[1] };			
			vec3 edgeAvec = cp.vertices[edgeA.x] - cp.vertices[edgeA.y];
			vec3 edgeBvec = cp.vertices[edgeB.x] - cp.vertices[edgeB.y];

			face.normal = glm::normalize(glm::cross(edgeAvec, edgeBvec));
			
		}


		cp.edges.resize(edgeSet.size());
		std::copy(edgeSet.begin(), edgeSet.end(), cp.edges.begin());


		return cp;


	}



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

}
