
#include "BatteryLibDef.h"
#include "TriangleMesh.h"

using namespace blib;

const float pi = static_cast<float>(std::acos(-1.0));

Eigen::Vector3f sphToCart(float a, float b) {
	return {
		cos(a) * sin(b),
		sin(a) * sin(b),
		cos(b)
	};
}


TriangleMesh blib::generateSphere(float radius, size_t polarSegments, size_t azimuthSegments)
{

	TriangleMesh mesh;
	
	const float stepX = (1.0f / polarSegments) * pi * 2.0f;
	const float stepY = (1.0f / azimuthSegments) * pi;

	for (int j = 0; j < azimuthSegments; j++) {
		const float b0 = j * stepY;
		const float b1 = (j + 1) * stepY;
		for (int i = 0; i < polarSegments; i++) {
			const float a0 = i * stepX;
			const float a1 = (i + 1) * stepX;
			
			const Eigen::Vector3f v[4] = {
				radius * sphToCart(a0, b0),
				radius * sphToCart(a0, b1),
				radius * sphToCart(a1, b0),
				radius * sphToCart(a1, b1)
			};			

			mesh.push_back({ v[0],v[1],v[3] });
			mesh.push_back({ v[0],v[3],v[2] });
		}
	}

	return mesh;
}
