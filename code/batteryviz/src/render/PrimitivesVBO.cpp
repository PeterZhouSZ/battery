#include "PrimitivesVBO.h"

VertexBuffer<VertexData> getQuadVBO()
{
	VertexBuffer<VertexData> vbo;

	std::vector<VertexData> data = {
		VertexData({ -1.0,-1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0,-1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0,1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0,1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 })
	};
	vbo.setData(data.begin(), data.end());
	vbo.setPrimitiveType(GL_QUADS);

	return vbo;
}

VertexBuffer<VertexData> getCubeVBO()
{
	VertexBuffer<VertexData> ivbo;


	const std::vector<VertexData> data = {
		VertexData({ -1.0f, -1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, -1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, 1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0f, 1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0f, -1.0f, -1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, -1.0f, -1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, 1.0f, -1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0f, 1.0f, -1.0 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 })
	};


	const std::vector<unsigned int> indices = {
		0, 1, 2, 2, 3, 0,
		3, 2, 6, 6, 7, 3,
		7, 6, 5, 5, 4, 7,
		4, 0, 3, 3, 7, 4,
		0, 5, 1, 5, 0, 4,
		1, 5, 6, 6, 2, 1
	};


	std::vector<VertexData> tridata;

	for (auto i = 0; i < indices.size(); i += 3) {
		auto a = indices[i];
		auto b = indices[i + 1];
		auto c = indices[i + 2];
		tridata.push_back(data[a]);
		tridata.push_back(data[b]);
		tridata.push_back(data[c]);
	}


	ivbo.setData(tridata.begin(), tridata.end());
	//ivbo.setIndices<unsigned int>(indices.begin(), indices.end(), GL_UNSIGNED_INT);
	ivbo.setPrimitiveType(GL_TRIANGLES);

	return ivbo;
}

