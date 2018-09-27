#include "VolumeGenerator.h"
#include "RandomGenerator.h"

#include <glm/gtx/norm.hpp>

#include <iostream>

namespace blib {


	RNGUniformFloat uniformDist(0, 1);

	float pointToNormBoxDistance(const vec3 & pt){		
		float x = glm::min(pt.x, 1.0f - pt.x);
		float y = glm::min(pt.y, 1.0f - pt.y);
		float z = glm::min(pt.z, 1.0f - pt.z);
		return glm::min(x, glm::min(y, z));
	}

	template<bool FLOOR = true>
	ivec3 normToDiscrete(const ivec3 & dim, const vec3 & normpos) {
		if (FLOOR) {
			return {
				static_cast<int>((dim.x - 1) * normpos.x),
				static_cast<int>((dim.y - 1) * normpos.y),
				static_cast<int>((dim.z - 1) * normpos.z),
			};
		}
		else {
			return {
				static_cast<int>(glm::ceil((dim.x - 1) * normpos.x)),
				static_cast<int>(glm::ceil((dim.y - 1) * normpos.y)),
				static_cast<int>(glm::ceil((dim.z - 1) * normpos.z)),
			};
		}
	}

	vec3 discreteToNorm(const ivec3 & dim, const ivec3 & discretePos) {
		return {
			static_cast<float>(discretePos.x) / dim.x,
			static_cast<float>(discretePos.y) / dim.y,
			static_cast<float>(discretePos.z) / dim.z,
		};
	}
	

	BLIB_EXPORT VolumeChannel generateSpheres(ivec3 res, const GeneratorSphereParams & p, bool * success)
	{
		VolumeChannel c(res, TYPE_UCHAR, false, "Spheres");


		if (p.withinBounds) {
			assert(p.rmin < 0.5f);
		}

		auto & ptr = c.getCurrentPtr();
		ptr.allocCPU();
		
		
		
		const uchar OCCUPIED = 255;
		const uchar EMPTY = 0;
		uchar * data = static_cast<uchar *>(ptr.getCPU());
		memset(data, EMPTY, res.x*res.y*res.z * sizeof(uchar));
		


		
		int tries = 0;
		

		float rrange = p.rmax - p.rmin;
		if (rrange < 0.0f) rrange = -rrange;

		
		vec3 posMax = { 1,1,1 };
		vec3 posMin = { 0,0,0 };
		
		if (p.withinBounds) {
			posMin += p.rmin;
			posMax -= p.rmin;
		}

		vec3 posRange = posMax - posMin;

		
		struct Sphere {
			vec3 pos;
			float r;
			float r2;
		};

		auto collision = [](const Sphere & a, const Sphere & b) -> bool{
			return glm::length2(a.pos - b.pos) < (a.r + b.r) * (a.r + b.r);

		};

		std::vector<Sphere> spheres;

		while (spheres.size() < p.N && tries < p.maxTries) {

			Sphere s;
			s.pos = vec3(uniformDist.next() * posRange.x, uniformDist.next() * posRange.y, uniformDist.next() * posRange.z) + posMin;			

			if (p.withinBounds) {
				float thisMaxR = glm::min(pointToNormBoxDistance(s.pos), p.rmax);			
				s.r = (uniformDist.next() * (thisMaxR - p.rmin)) + p.rmin;
			}
			else {
				s.r = (uniformDist.next() * rrange) + p.rmin;
			}		

			if(!p.overlapping){				
				bool isOk = true;
				for (auto & other : spheres) {					
					if (collision(s, other)) {
						isOk = false;
						break;
					}
				}

				if (!isOk) {
					tries++;
					continue;
				}

			}


			s.r2 = s.r*s.r;
			spheres.push_back(s);
		}

		if (tries >= p.maxTries) {

			*success = false;
			return c;
		}

		

		//Rasterize
		for (auto & s : spheres) {
			ivec3 boundMin = normToDiscrete<true>(res, s.pos - vec3(s.r));
			ivec3 boundMax = normToDiscrete<true>(res, s.pos + vec3(s.r));

			if (p.withinBounds) {
				assert(boundMin.x >= 0 && boundMin.y >= 0 && boundMin.z >= 0);
				assert(boundMax.x <= res.x && boundMax.y <= res.y && boundMin.z <= res.z);
			}

			boundMin = glm::clamp(boundMin, ivec3(0), res - ivec3(1));
			boundMax = glm::clamp(boundMax, ivec3(0), res - ivec3(1));

			for (auto z = boundMin.z; z <= boundMax.z; z++) {
				for (auto y = boundMin.y; y <= boundMax.y; y++) {
					for (auto x = boundMin.x; x <= boundMax.x; x++) {
						vec3 pos = discreteToNorm(res, { x,y,z });

						if (glm::length2(pos - s.pos) < s.r2) {
							data[linearIndex(res, { x,y,z })] = OCCUPIED;
						}

					}
				}
			}

		
		}

		
		float totalArea = 0.0;
		float totalVolume = 0.0f;
		for (auto & s : spheres) {
			totalArea += 4.0f * glm::pi<float>() * s.r2;
			totalVolume += (4.0f / 3.0f) * glm::pi<float>() * s.r2 * s.r;		
		}

		if (p.withinBounds) {
			std::cout << "[ANALYTIC] Total surface area: " << totalArea << ", volume: " << totalVolume << " porosity: " << 1.0 - totalVolume << std::endl;
		}



		ptr.commit();

		*success = true;
		return c;

	}

}