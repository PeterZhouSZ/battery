#include "GeometryIO.h"

#include <fstream>
#include <set>
#include <algorithm>

namespace blib {


	BLIB_EXPORT blib::TriangleMesh blib::loadParticleMesh(const std::string & path)
	{
		using Edge = blib::TriangleMesh::Edge;
		using Face = blib::TriangleMesh::Face;

		blib::TriangleMesh m;
		

		std::ifstream f(path);

		if (!f.good())
			throw "ConvexPolyhedron::loadFromFile invalid file";		


		int nv, nf;
		f >> nv;

		if (nv <= 0)
			throw "ConvexPolyhedron::loadFromFile invalid number of vertices";

		m.vertices.resize(nv);

		for (auto i = 0; i < nv; i++) {
			f >> m.vertices[i].x >> m.vertices[i].y >> m.vertices[i].z;
		}

		f >> nf;

		if (nf <= 0)
			throw "ConvexPolyhedron::loadFromFile invalid number of faces";

		m.faces.resize(nf);

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

			Face & face = m.faces[i];

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



		}



		m.edges.resize(edgeSet.size());
		std::copy(edgeSet.begin(), edgeSet.end(), m.edges.begin());

		m.recomputeNormals();

		return m;
	}
}

