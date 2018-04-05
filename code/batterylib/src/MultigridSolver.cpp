#include "MultigridSolver.h"

using namespace blib;

template class MultigridSolver<float>;
template class MultigridSolver<double>;


#include "JacobiSolver.h"
#include <cassert>
#include "../../batteryviz/src/utility/IOUtility.h"
#include <numeric>
#include <array>

#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include<Eigen/SparseCholesky>	

//#define MG_LINSYS_TO_FILE


#ifdef MG_LINSYS_TO_FILE
	#include <fstream>
#endif

//#define PERIODIC
//#define HARMONIC
#define MOLENAAR

template <typename T = double>
void addDebugChannel(Volume & vol,  const Eigen::Matrix<T, Eigen::Dynamic, 1> & v, ivec3 dim, const std::string & name, int levnum = -1, bool normalize = false) {
		
	auto & c = vol.getChannel(vol.addChannel(dim, TYPE_FLOAT));
	


	Eigen::Matrix<float, Eigen::Dynamic, 1> tmp = v.template cast<float>();
	

	/*if (std::is_same<float, T>::value) {
		tmp = v;
	}
	else {
		tmp = ;
	}
	*/


	if (normalize) {
		tmp = tmp.cwiseAbs();
		auto maxc = tmp.maxCoeff();
		auto minc = tmp.minCoeff();
		tmp = (tmp.array() - minc) / (maxc - minc);
		//tmp *= T(10);
	}

	memcpy(c.getCurrentPtr().getCPU(), tmp.data(), dim.x * dim.z* dim.y * sizeof(float));
	c.getCurrentPtr().commit();

	char buf[24]; itoa(levnum, buf, 10);
	c.setName(name + buf);


}


template <typename T>
MultigridSolver<T>::MultigridSolver(bool verbose)
	: _verbose(verbose)
{
	
	_restrictOp = std::array<T, 27>{
		0,1,0 , 1,2,1 , 0,1,0,
		1,2,1 , 2,4,2 , 1,2,1,
		0,1,0 , 1,2,1 , 0,1,0
	};

	_interpOp = std::array<T, 27>{
		1,1,1, 2,4,2, 1,2,1,
		2,4,2, 4,8,4, 2,4,2,
		1,2,1, 2,4,2, 1,2,1
	};

	
}

template <typename T, typename K>
void conv3D(
	K * src, ivec3 srcDim,
	K * dest, ivec3 destDim,
	const std::array<T,27> & kernel
) {
	
	auto srcStride = { 1, srcDim.x, srcDim.x * srcDim.y };

	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {

				auto destI = linearIndex(destDim, { x,y,z });			

				K val = K(0);
				T wsum = 0.0;
				
				for (auto k = 0; k < 3; k++) {
					for (auto j = 0; j < 3; j++) {
						for (auto i = 0; i < 3; i++) {				


							const T & w = kernel[i + 3 * j + k * 9];
							if (w == T(0)) continue;

							ivec3 srcPos = 2 * ivec3(x, y, z) + ivec3(i - 1, j - 1, k - 1);
							if (!isValidPos(srcDim, srcPos)) continue;

							/*if (srcPos.x < 0) {
								val += w * 1.0f;
								wsum += w;
								continue;
							}
							else if (srcPos.x >= srcDim.x) {
								val += w * 0.0f;
								wsum += w;
								continue;
							}

							if (srcPos.y < 0) {
								srcPos.y = 0;
							}
							if (srcPos.z < 0) {
								srcPos.z = 0;
							}
							if (srcPos.y >= srcDim.y) {
								srcPos.y = srcDim.y - 1;
							}
							if (srcPos.z >= srcDim.z) {
								srcPos.z = srcDim.z - 1;
							}*/

							auto srcI = linearIndex(srcDim, srcPos);
							val += w * src[srcI];
							wsum += w;
						}
					}
				}

				if (wsum > T(0))
					dest[destI] = val / wsum;
				else
					dest[destI] = T(0);

			}
		}
	}
}

template <typename T>
void pointRestriction(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim 	
) {
	assert(srcDim.x == destDim.x * 2);
	assert(srcDim.y == destDim.y * 2);
	assert(srcDim.z == destDim.z * 2);

	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				ivec3 ipos = ivec3(x, y, z);
				auto srcI = linearIndex(srcDim, 2 * ipos);
				auto destI = linearIndex(destDim, ipos);
				dest[destI] = src[srcI];
			}
		}
	}
}

template <typename T>
void restriction(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim
) {
	/*assert(srcDim.x == destDim.x * 2);
	assert(srcDim.y == destDim.y * 2);
	assert(srcDim.z == destDim.z * 2);*/
	
	
	//#pragma omp parallel for if(destDim.x > 4)
	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				ivec3 ipos = { x,y,z };
				ivec3 iposSrc = ipos * 2;
				auto srcI = linearIndex(srcDim, iposSrc);
				auto destI = linearIndex(destDim, ipos);

				ivec3 s = { 1, srcDim.x, srcDim.x * srcDim.y };
				if (iposSrc.x == srcDim.x - 1 /*|| x == 0*/) s[0] = 0;
				if (iposSrc.y == srcDim.y - 1 /*|| y == 0*/) s[1] = 0;
				if (iposSrc.z == srcDim.z - 1 /*|| z == 0*/) s[2] = 0;

				T val = src[srcI] +
					src[srcI + s[0]] +
					src[srcI + s[1]] +
					src[srcI + s[1] + s[0]] +
					src[srcI + s[2]] +
					src[srcI + s[2] + s[0]] +
					src[srcI + s[2] + s[1]] +
					src[srcI + s[2] + s[1] + s[0]];
				
				val *= T(1.0 / 8.0);

				dest[destI] = val;
			}
		}
	}



}


ivec3 neighDirs[8] = {
	{ 0,0,0 },
	{ 1,0,0 },
	{ 0,1,0 },
	{ 1,1,0 },
	{ 0,0,1 },
	{ 1,0,1 },
	{ 0,1,1 },
	{ 1,1,1 }
};

template <typename T>
void restrictionWeighted(
	Dir dir,
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim,
	T * srcWeights
) {
#ifndef DEBUG
#pragma omp parallel for if(destDim.x > 4)
#endif
	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				ivec3 ipos = { x,y,z };
				ivec3 iposSrc = ipos * 2;
				auto srcI = linearIndex(srcDim, iposSrc);
				auto destI = linearIndex(destDim, ipos);

#ifdef PERIODIC
				T val = 0;
				T W = 0;
				for (auto d = 0; d < 8; d++) {					
					ivec3 newPos = (ipos + neighDirs[d]) % srcDim;										
					auto newI = linearIndex(srcDim,newPos);
					val += src[newI] * srcWeights[newI];
					W += srcWeights[newI];					
				}
#else

				ivec3 s = { 1, srcDim.x, srcDim.x * srcDim.y };
				
				if (iposSrc.x == srcDim.x - 1 /*|| x == 0*/) {
					s[0] = 0;
				}
				if (iposSrc.y == srcDim.y - 1 /*|| y == 0*/) {
					s[1] = 0;
				}
				if (iposSrc.z == srcDim.z - 1 /*|| z == 0*/) {
					s[2] = 0;
				}


				T vals[8] = {
					src[srcI],
					src[srcI + s[0]],
					src[srcI + s[1]],
					src[srcI + s[1] + s[0]],
					src[srcI + s[2]],
					src[srcI + s[2] + s[0]],
					src[srcI + s[2] + s[1]],
					src[srcI + s[2] + s[1] + s[0]]
				};

				T w[8] = {
					srcWeights[srcI],
					srcWeights[srcI + s[0]],
					srcWeights[srcI + s[1]],
					srcWeights[srcI + s[1] + s[0]],
					srcWeights[srcI + s[2]],
					srcWeights[srcI + s[2] + s[0]],
					srcWeights[srcI + s[2] + s[1]],
					srcWeights[srcI + s[2] + s[1] + s[0]]
				};

				T val = 0;
				T W = 0;
				for (auto i = 0; i < 8; i++) {					
					val += vals[i] * w[i];
					W += w[i];
				}

				/*T _val = 0;
				T _W= 0;
				for (auto i = 0; i < 8; i++) {
					if (dir == X_NEG && iposSrc.x == srcDim.x - 1 && (i % 2 == 1)) {
						continue;
					}
					_val += vals[i] * w[i];
					_W += w[i];
				}*/

				/*T val = src[srcI] * srcWeights[srcI] +
					src[srcI + s[0]] * srcWeights[srcI + s[0]] +
					src[srcI + s[1]] * srcWeights[srcI + s[1]] +
					src[srcI + s[1] + s[0]] * srcWeights[srcI + s[0] + s[1]] +
					src[srcI + s[2]] * srcWeights[srcI + s[2]] +
					src[srcI + s[2] + s[0]] * srcWeights[srcI + s[2] + s[0]] +
					src[srcI + s[2] + s[1]] * srcWeights[srcI + s[2] + s[1]] +
					src[srcI + s[2] + s[1] + s[0]] * srcWeights[srcI + s[2] + s[1] + s[0]];*/

				/*T W = srcWeights[srcI] + srcWeights[srcI + s[0]] + srcWeights[srcI + s[1]] + srcWeights[srcI + s[0] + s[1]] + srcWeights[srcI + s[2]] +
					srcWeights[srcI + s[2] + s[0]] + srcWeights[srcI + s[2] + s[1]] + srcWeights[srcI + s[2] + s[1] + s[0]];*/
#endif

				val /= W;
				//_val /= _W;
				//val *= T(1.0 / 8.0);

				dest[destI] = val;
			}
		}
	}
}
/*

template <typename T>
void restrictionWeightedPlane(
	T * src, ivec3 srcDim, int dir, int plane //dir .. which dimension, plane - which plane
	T * dest, ivec3 destDim,
	T * srcWeights
) {
#pragma omp parallel for if(destDim.x > 4)
	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				ivec3 ipos = { x,y,z };
				ivec3 iposSrc = ipos * 2;
				auto srcI = linearIndex(srcDim, iposSrc);
				auto destI = linearIndex(destDim, ipos);

				ivec3 s = { 1, srcDim.x, srcDim.x * srcDim.y };
				if (iposSrc.x == srcDim.x - 1 / *|| x == 0* /) s[0] = 0;
				if (iposSrc.y == srcDim.y - 1 / *|| y == 0* /) s[1] = 0;
				if (iposSrc.z == srcDim.z - 1 / *|| z == 0* /) s[2] = 0;

				T val = src[srcI] * srcWeights[srcI] +
					src[srcI + s[0]] * srcWeights[srcI + s[0]] +
					src[srcI + s[1]] * srcWeights[srcI + s[1]] +
					src[srcI + s[1] + s[0]] * srcWeights[srcI + s[0] + s[1]] +
					src[srcI + s[2]] * srcWeights[srcI + s[2]] +
					src[srcI + s[2] + s[0]] * srcWeights[srcI + s[2] + s[0]] +
					src[srcI + s[2] + s[1]] * srcWeights[srcI + s[2] + s[1]] +
					src[srcI + s[2] + s[1] + s[0]] * srcWeights[srcI + s[2] + s[1] + s[0]];

				T W = srcWeights[srcI] + srcWeights[srcI + s[0]] + srcWeights[srcI + s[1]] + srcWeights[srcI + s[0] + s[1]] + srcWeights[srcI + s[2]] +
					srcWeights[srcI + s[2] + s[0]] + srcWeights[srcI + s[2] + s[1]] + srcWeights[srcI + s[2] + s[1] + s[0]];


				val /= W;
				//val *= T(1.0 / 8.0);

				dest[destI] = val;
			}
		}
	}
}*/

template <typename T>
void interpolation(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim //higher	
) {

#ifndef DEBUG
	#pragma omp parallel for if(destDim.x > 4)
#endif
	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {

				ivec3 ipos = { x,y,z };
				ivec3 iposSrc = ipos / 2;
				ivec3 r = ivec3(ipos.x % 2, ipos.y % 2, ipos.z % 2 )*2 - 1;
				auto destI = linearIndex(destDim, ipos);
				auto srcI = linearIndex(srcDim, iposSrc);

				ivec3 srcStride = { 1, srcDim.x, srcDim.x * srcDim.y };
				if (x == destDim.x - 1 || x == 0) srcStride[0] = 0;
				if (y == destDim.y - 1 || y == 0) srcStride[1] = 0;
				if (z == destDim.z - 1 || z == 0) srcStride[2] = 0;

				T val = T(3.0 / 4.0) * src[srcI] + T(1.0 / 12.0) * (
					src[srcI + srcStride[0] * r[0]] + src[srcI + srcStride[1] * r[1]] + src[srcI + srcStride[2] * r[2]]);


				dest[destI] = val;

			}
		}
	}


}

template <typename T>
void interpolationWeighted(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim, //higher	
	T * srcWeights
) {

#ifndef DEBUG
#pragma omp parallel for if(destDim.x > 4)
#endif
	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {

				ivec3 ipos = { x,y,z };
				ivec3 iposSrc = ipos / 2;
				//Direction
				// -1 for even, 1 for odd 
				ivec3 r = ivec3(ipos.x % 2, ipos.y % 2, ipos.z % 2) * 2 - 1; 
				auto destI = linearIndex(destDim, ipos);
				auto srcI = linearIndex(srcDim, iposSrc);



				ivec3 srcStride = r * ivec3( 1, srcDim.x, srcDim.x * srcDim.y );
				if (x == destDim.x - 1 || x == 0) srcStride[0] = 0;
				if (y == destDim.y - 1 || y == 0) srcStride[1] = 0;
				if (z == destDim.z - 1 || z == 0) srcStride[2] = 0;

#ifdef PERIODIC
				T vals[8];
				T w[8];

				
				for (auto d = 0; d < 8; d++) {
					ivec3 step = r * neighDirs[d];
					ivec3 newPos = (ipos + step + srcDim) % srcDim;
					auto newI = linearIndex(srcDim, newPos);
					vals[d] = src[newI];
					w[d] = srcWeights[newI];
				}

	/*			T vals = src[srcI] * srcWeights[srcI];
				T W = srcWeights[srcI];

				for (auto d = 0; d < 3; d++) {
					for (auto sgn = -1; sgn <= 1; sgn += 2) {
						ivec3 newPos = ipos;
						newPos[d] = (newPos[d] + sgn + srcDim[d]) % srcDim[d];
						auto newI = linearIndex(srcDim, newPos);
						val += src[newI] * srcWeights[newI];
						W += srcWeights[newI];
					}
				}*/
#else
				T vals[8] = {
					src[srcI],
					src[srcI	+ srcStride[0]],
					src[srcI					+ srcStride[1]],
					src[srcI	+ srcStride[0]	+ srcStride[1]],
					src[srcI									+ srcStride[2]],
					src[srcI	+ srcStride[0]					+ srcStride[2]],
					src[srcI					+ srcStride[1]	+ srcStride[2]],
					src[srcI	+ srcStride[0]	+ srcStride[1]	+ srcStride[2]]
				};

				/*if (x == destDim.x - 1 || x == 0) srcStride[0] = 0;
				if (y == destDim.y - 1 || y == 0) srcStride[1] = 0;
				if (z == destDim.z - 1 || z == 0) srcStride[2] = 0;*/

				T w[8] = {
					srcWeights[srcI],
					srcWeights[srcI + srcStride[0]],
					srcWeights[srcI + srcStride[1]],
					srcWeights[srcI + srcStride[0] + srcStride[1]],
					srcWeights[srcI + srcStride[2]],
					srcWeights[srcI + srcStride[0] + srcStride[2]],
					srcWeights[srcI + srcStride[1] + srcStride[2]],
					srcWeights[srcI + srcStride[0] + srcStride[1] + srcStride[2]]
				};
				


				/*if (x == 0 || x == destDim.x - 1) {
					vals[1] = 0;
					vals[3] = 0;
					vals[5] = 0;
					vals[7] = 0;

					w[1] = 0;
					w[3] = 0;
					w[4] = 0;
					w[7] = 0;
				}

				if (y == 0 || y == destDim.y - 1) {
					vals[2] = 0;
					vals[3] = 0;
					vals[6] = 0;
					vals[7] = 0;

					w[2] = 0;
					w[3] = 0;
					w[6] = 0;
					w[7] = 0;
				}

				if (z == 0 || z == destDim.y - 1) {
					vals[4] = 0;
					vals[5] = 0;
					vals[6] = 0;
					vals[7] = 0;

					w[4] = 0;
					w[5] = 0;
					w[6] = 0;
					w[7] = 0;
				}*/

#endif
				


				
				const T a = T(1.0 / 4.0);
				const T ainv = T(1.0) - a;
				
				
				

				

				


				T w_x0y0 = ainv * w[0] + a * w[4];
				T w_x0y1 = ainv * w[2] + a * w[6];
				T w_x0 = ainv * w_x0y0 + a * w_x0y1;

				T w_x1y0 = ainv * w[1] + a * w[5];
				T w_x1y1 = ainv * w[3] + a * w[7];
				T w_x1 = ainv * w_x1y0 + a * w_x1y1;

				T w_val = ainv * w_x0 + a * w_x1;


				T x0y0 = ainv * vals[0] * w[0] + a * vals[4] * w[4];
				T x0y1 = ainv * vals[2] * w[2] + a * vals[6] * w[6];
				T x0 = ainv * x0y0 + a * x0y1;

				T x1y0 = ainv * vals[1] * w[1] + a * vals[5] * w[5];
				T x1y1 = ainv * vals[3] * w[3] + a * vals[7] * w[7];
				T x1 = ainv * x1y0 + a * x1y1;

				T val = ainv * x0 + a * x1;
				val /= w_val;



/*

				T old  = T(3.0 / 4.0) * src[srcI] + T(1.0 / 12.0) * (
					src[srcI + srcStride[0]] +
					src[srcI + srcStride[1]] + 
					src[srcI + srcStride[2]]
					);*/


				dest[destI] = val;

				char b;
				b = 0;

			}
		}
	}


}



template <typename T>
void pointInterpolation(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim //higher	
) {

	

	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				ivec3 ipos = { x,y,z };
				ivec3 iposSrc = ipos / 2;
				auto destI = linearIndex(destDim, ipos);
				auto srcI = linearIndex(srcDim, iposSrc);

				ivec3 srcStride = { 1, srcDim.x, srcDim.x * srcDim.y };
				if (x == destDim.x - 1) srcStride[0] = 0;
				if (y == destDim.y - 1) srcStride[1] = 0;
				if (z == destDim.z - 1) srcStride[2] = 0;

				const ivec3 mods = { x % 2, y % 2, z % 2 };
				const int modSum = mods.x + mods.y + mods.z;


				T val = T(0);
				
				//Colocated				
				if (modSum == 0) {					
					val = src[srcI];
				}
				//Edge interpolation
				else if (modSum == 1) {
					int index = (mods[0] == 1) ? 0 : ((mods[1] == 1) ? 1 : 2);
					
					ivec3 dx = ivec3(0);
					dx[index] = 1;

					//assert(srcI + srcStride[index] == linearIndex(srcDim, iposSrc + dx));

					val = (
						src[srcI] +
						src[srcI + srcStride[index]]
						) * T(1.0 / 2.0);


				}
				//Face interpolation
				else if (modSum == 2) {
					int coIndex = (mods[0] == 0) ? 0 : ((mods[1] == 0) ? 1 : 2);
					int i0 = (coIndex + 1) % 3;
					int i1 = (coIndex + 2) % 3;		

					val = (
						src[srcI] +
						src[srcI + srcStride[i1]] +
						src[srcI + srcStride[i0]] +
						src[srcI + srcStride[i0] + srcStride[i1]]
						) * T(1.0 / 4.0);
				}
				//Cube interpolation  //modSum == 3
				else {
					val = (
						src[srcI] +
						src[srcI + srcStride[0]] +
						src[srcI + srcStride[1]] +
						src[srcI + srcStride[0] + srcStride[1]] +
						src[srcI + srcStride[2]] +
						src[srcI + srcStride[1] + srcStride[2]] +
						src[srcI + srcStride[0] + srcStride[2]] +
						src[srcI + srcStride[0] + srcStride[1] + srcStride[2]]
						) * T(1.0 / 8.0);					
				}


				dest[destI] = val;				

			}
		}
	}

}

template <typename T>
void interp3D(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim //higher	
) {
		

	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				auto destI = linearIndex(destDim, { x,y,z });

				ivec3 srcStride = { 1, srcDim.x, srcDim.x * srcDim.y };
				if (x == destDim.x-1) srcStride[0] = 0;
				if (y == destDim.y-1) srcStride[1] = 0;
				if (z == destDim.z-1) srcStride[2] = 0;


				vec3 sPos = vec3(x, y, z) * 0.5f;
				ivec3 ipos = ivec3(sPos);
				vec3 fract = glm::fract(sPos);

				auto srcI = linearIndex(srcDim, ipos);

				assert(srcI >= 0 && srcI < srcDim.x * srcDim.x * srcDim.y);
							
				T vals[8] = {
					src[srcI],														//0
					src[srcI + srcStride[0]],										//1
					src[srcI +					srcStride[1]],						//2
					src[srcI + srcStride[0] +	srcStride[1]],						//3
					src[srcI									+ srcStride[2]],	//4
					src[srcI + srcStride[0]						+ srcStride[2]],	//5
					src[srcI +					srcStride[1]	+ srcStride[2]],	//6
					src[srcI + srcStride[0] +	srcStride[1]	+ srcStride[2]]		//7
				};

				T x0 = glm::mix(
					glm::mix(vals[0], vals[2], fract.y), //0,0,0 vs 0,1,0
					glm::mix(vals[4], vals[6], fract.y), //0,0,1 vs 0,1,1
					fract.z
				);
				T x1 = glm::mix(
					glm::mix(vals[1], vals[3], fract.y), //1,0,0 vs 1,1,0
					glm::mix(vals[5], vals[7], fract.y), //1,0,1 vs 1,1,1
					fract.z
				);

				float val = glm::mix(x0, x1, fract.x);

				dest[destI] = val;

			}
		}
	}



}


template <typename T>
void interpZeroOrder(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim
)
{


	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				auto destI = linearIndex(destDim, { x,y,z });

				ivec3 srcStride = { 1, srcDim.x, srcDim.x * srcDim.y };
				if (x == destDim.x - 1) srcStride[0] = 0;
				if (y == destDim.y - 1) srcStride[1] = 0;
				if (z == destDim.z - 1) srcStride[2] = 0;


				vec3 sPos = vec3(x, y, z) * 0.5f;
				ivec3 ipos = ivec3(sPos);
				vec3 fract = glm::fract(sPos);

				auto srcI = linearIndex(srcDim, ipos);

				assert(srcI >= 0 && srcI < srcDim.x * srcDim.x * srcDim.y);

				T vals[8] = {
					src[srcI],														//0
					src[srcI + srcStride[0]],										//1
					src[srcI + srcStride[1]],						//2
					src[srcI + srcStride[0] + srcStride[1]],						//3
					src[srcI + srcStride[2]],	//4
					src[srcI + srcStride[0] + srcStride[2]],	//5
					src[srcI + srcStride[1] + srcStride[2]],	//6
					src[srcI + srcStride[0] + srcStride[1] + srcStride[2]]		//7
				};		
				
				T val = T(0);
				for (auto k = 0; k < 8; k++) {
					val += vals[k];
				}
				val /= 8;				

				dest[destI] = val;

			}
		}
	}
}


template <typename T>
bool MultigridSolver<T>::prepare(
	Volume & v, 
	const uchar * Dmask, ivec3 origDim, Dir dir, T d0, T d1,
	uint levels,
	vec3 cellDim
){

	_cellDim = cellDim;
	_iterations = 0;
	_dir = dir;
	bool res = true;

	_lv = levels;

	//Prepare lin. system for all levels
	_A.resize(_lv);
	_f.resize(_lv);
	_x.resize(_lv);
	_tmpx.resize(_lv);
	_r.resize(_lv);
	_dims.resize(_lv);

	//Generate D volume for smaller resolutions
	_D.resize(_lv);
	//std::vector<std::vector<T>> Dlevels(_lv);
	//std::vector<std::vector<float>> Dlevels_interp(_lv);
	
	

	const size_t origTotal = origDim.x * origDim.y * origDim.z;

	size_t cntd0 = 0;
	for (auto i = 0; i < origTotal; i++) {
		if (Dmask[i] == 0) cntd0++;
	}
	_porosity = cntd0 / T(origTotal);

	//Generate first level of D
	{
		_D.front().resize(origTotal);
		_dims.front() = origDim;

		for (auto i = 0; i < origTotal; i++) {
			_D.front()[i] = (Dmask[i] == 0) ? d0 : d1;
		}

		res &= prepareAtLevelFVM(_D[0].data(), _dims[0], dir, 0);


		//Dlevels_interp[0].resize(origTotal, 0.0f);
	}

	//Generate rest of levels, based on higher resolution
	for (uint i = 1; i < _lv; i++) {
		//const int divFactor = (1 << i);
		ivec3 dim = _dims[i-1] / 2;
		if (_dims[i - 1].x % 2 == 1) dim.x++;
		if (_dims[i - 1].y % 2 == 1) dim.y++;
		if (_dims[i - 1].z % 2 == 1) dim.z++;

		ivec3 dtmp = (_dims[i - 1] + (_dims[i - 1] - ivec3(1))) / 2;
		

		const size_t total = dim.x * dim.y * dim.z;

		_dims[i] = dim;
		_D[i].resize(total);
		//conv3D(Dlevels[i - 1].data(), _dims[i - 1], Dlevels[i].data(), _dims[i], _restrictOp);
		restriction(_D[i - 1].data(), _dims[i - 1], _D[i].data(), _dims[i]);		
		for (auto & d : _D[i]) {
			d *= T(0.5);
		}
		
		
		//Dlevels_interp[i].resize(total, 0.0f);
		

		res &= prepareAtLevelFVM(_D[i].data(), _dims[i], dir, i);
	}




	for (uint i = 0; i < _lv; i++) {

		//addDebugChannel<T>(v, Dlevels[i].data(), _dims[i], "D ", i, false);

		

		/*if(i < _lv - 1){
			interp3D(Dlevels[i + 1].data(), _dims[i + 1], Dlevels_interp[i].data(), _dims[i]);		
			auto id = v.addChannel(_dims[i], TYPE_FLOAT);
			auto & c = v.getChannel(id);
			memcpy(c.getCurrentPtr().getCPU(), Dlevels_interp[i].data(), _dims[i].x * _dims[i].y * _dims[i].z * sizeof(float));
			c.getCurrentPtr().commit();
		}*/



	}

	for (auto i = 0; i < _lv; i++) {
		_r[i].resize(_x[i].size());
		_tmpx[i].resize(_x[i].size());
	}

	if (_verbose) {
		std::cout << "Prepared multigrid with L = " << _lv << " res: " << origDim.x << ", "<< origDim.y << ", " << origDim.z << std::endl;
	}
	
	return res;
}


/*
template <typename T>
void galerkin(MultigridSolver<T>::SparseMat & A) {



}*/


template <typename T>
bool MultigridSolver<T>::prepareAtLevelFVM(
	const T * D, ivec3 dim, Dir dir,
	const uint level
) {

	using vec3 = glm::tvec3<T, glm::highp>;
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);

	uint dirPrimary = getDirIndex(dir);
	uint dirSecondary[2] = { (dirPrimary + 1) % 3, (dirPrimary + 2) % 3 };
	const T highConc = 1.0f;
	const T lowConc = 0.0f;
	T concetrationBegin = (getDirSgn(dir) == 1) ? highConc : lowConc;
	T concetrationEnd = (getDirSgn(dir) == 1) ? lowConc : highConc;

	if (level > 0) {
		concetrationEnd = 0;
		concetrationBegin = 0;
	}


	size_t N = dim.x * dim.y * dim.z;
	size_t M = N;

	_f[level].resize(M);
	_x[level].resize(N);
	_A[level].resize(M, N);
	_A[level].reserve(Eigen::VectorXi::Constant(N, 7));


	
	//D getter

	const auto getD = [&dim, D](ivec3 pos) {
		return D[linearIndex(dim, pos)];
	};

	/*const auto getDFine = [&_dims,_D,level](ivec3 pos) {
		auto dims = _dims.front();
		auto & D = _D.front();
			
		return D[linearIndex(dim, pos)];
	};*/



	
#ifdef PERIODIC
	const auto sample = [&getD, dim, D](ivec3 pos, Dir dir) {
		const int k = getDirIndex(dir);
		ivec3 newPos = pos;
		int sgn = getDirSgn(dir);
		newPos[k] = (newPos[k] + sgn + dim[k]) % dim[k];
		return getD(newPos);
	};
#else

	const auto sample = [&getD, dim, D](ivec3 pos, Dir dir) {
		const int k = getDirIndex(dir);
		assert(pos.x >= 0 && pos.y >= 0 && pos.z >= 0);
		assert(pos.x < dim.x && pos.y < dim.y && pos.z < dim.z);

		ivec3 newPos = pos;
		int sgn = getDirSgn(dir);
		if (pos[k] + sgn < 0 || pos[k] + sgn >= dim[k]) {
			//do nothing
		}
		else {
			newPos[k] += getDirSgn(dir);
		}
		return getD(newPos);
	};


#endif

	const vec3 faceArea = {
		_cellDim.y * _cellDim.z,
		_cellDim.x * _cellDim.z,
		_cellDim.x * _cellDim.y,
	};


	
	struct Coeff {
		Dir dir;
		T val;
		signed long col;
		bool useInMatrix;
		bool operator < (const Coeff & b) const { return this->col < b.col; }
	};



	ivec3 stride = { 1, dim.x, dim.x*dim.y };

	for (auto z = 0; z < dim.z; z++) {
		for (auto y = 0; y < dim.y; y++) {
			for (auto x = 0; x < dim.x; x++) {


				const ivec3 ipos = { x,y,z };
				auto i = linearIndex(dim, ipos);

				T Di = getD(ipos);
				

#ifdef HARMONIC
				
				vec3 Dneg = vec3(
					sample(ipos, X_NEG),
					sample(ipos, Y_NEG),
					sample(ipos, Z_NEG)
				);
				Dneg.x = (2 * Dneg.x * Di) / (Dneg.x + Di);
				Dneg.y = (2 * Dneg.y * Di) / (Dneg.y + Di);
				Dneg.z = (2 * Dneg.z * Di) / (Dneg.z + Di);

				vec3 Dpos = vec3(
					sample(ipos, X_POS),
					sample(ipos, Y_POS),
					sample(ipos, Z_POS)
				);
				Dpos.x = (2 * Dpos.x * Di) / (Dpos.x + Di);
				Dpos.y = (2 * Dpos.y * Di) / (Dpos.y + Di);
				Dpos.z = (2 * Dpos.z * Di) / (Dpos.z + Di);


#else
				auto Dvec = vec3(Di);
				auto Dneg = (vec3(
					sample(ipos, X_NEG),
					sample(ipos, Y_NEG),
					sample(ipos, Z_NEG)
				) + vec3(Dvec)) * T(0.5);
				


				auto Dpos = (vec3(
					sample(ipos, X_POS),
					sample(ipos, Y_POS),
					sample(ipos, Z_POS)
				) + vec3(Dvec)) * T(0.5);
#endif
				

				std::array<Coeff, 7> coeffs;
				T rhs = T(0);
				for (auto k = 0; k <= DIR_NONE; k++) {
					coeffs[k].dir = Dir(k);
					coeffs[k].val = T(0);
					coeffs[k].useInMatrix = true;
				}
				coeffs[DIR_NONE].col = i;



				//Calculate coeffs for all except diagonal
				for (auto j = 0; j < DIR_NONE; j++) {
					auto & c = coeffs[j];
					auto k = getDirIndex(c.dir);
					auto sgn = getDirSgn(c.dir);
					auto Dface = (sgn == -1) ? Dneg[k] : Dpos[k];

					/*
					Boundary defaults
					1. half size cell
					2. von neumann zero grad
					*/
					auto cellDist = _cellDim;
					c.useInMatrix = true;

#ifndef PERIODIC
					if (ipos[k] == 0 && sgn == -1) {
						cellDist[k] = _cellDim[k] * T(0.5);
						c.useInMatrix = false;
					}
					if (ipos[k] == dim[k] - 1 && sgn == 1) {
						cellDist[k] = _cellDim[k] * T(0.5);
						c.useInMatrix = false;
					}
#endif

					c.val = (Dface * faceArea[k]) / cellDist[k];

#ifdef PERIODIC
					auto newPos = ipos;
					newPos[k] = (newPos[k] + sgn + dim[k]) % dim[k];
					c.col = linearIndex(dim, newPos);
#else
					c.col = i + sgn * stride[k];
#endif

					//Add to diagonal
					if (c.useInMatrix || k == dirPrimary)
						coeffs[DIR_NONE].val -= c.val;
				}

				if (ipos[dirPrimary] == 0) {
					Dir dir = getDir(dirPrimary, -1);
					rhs -= coeffs[dir].val * concetrationBegin;
				}
				else if (ipos[dirPrimary] == dim[dirPrimary] - 1) {
					Dir dir = getDir(dirPrimary, 1);
					rhs -= coeffs[dir].val * concetrationEnd;
				}

				//Matrix coefficients
				std::sort(coeffs.begin(), coeffs.end());
				for (auto & c : coeffs) {
					if (c.useInMatrix) {
						_A[level].insert(i, c.col) = c.val;
					}
				}

				//Right hand side
				_f[level][i] = rhs;

				//initial guess
				if (getDirSgn(dir) == 1)
					_x[level][i] = 1.0f - (ipos[dirPrimary] / T(dim[dirPrimary] + 1));
				else
					_x[level][i] = (ipos[dirPrimary] / T(dim[dirPrimary] + 1));

			}
		}
	}
	
	_A[level].makeCompressed();	

	
#ifdef MG_LINSYS_TO_FILE


	{
		char buf[24]; itoa(level, buf, 10);
		std::ofstream f("A_" + std::string(buf) + ".dat");

		for (auto i = 0; i < _A[level].rows(); i++) {
			for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(_A[level], i); it; ++it) {
				auto  j = it.col();
				f << i << " " << j << " " << it.value() << "\n";
			}

			if (_A[level].rows() < 100 || i % (_A[level].rows() / 100))
				f.flush();
		}
	}

	{
		char buf[24]; itoa(level, buf, 10);
		std::ofstream f("B_" + std::string(buf) + ".txt");
		for (auto i = 0; i < _f[level].size(); i++) {
			f << _f[level][i] << "\n";
			if (_f[level].size() < 100 || i % (_f[level].size() / 100))
				f.flush();
		}

	}

	{
		char buf[24]; itoa(level, buf, 10);
		std::ofstream f("D_" + std::string(buf) + ".txt");
		for (auto i = 0; i < _D[level].size(); i++) {
			f << _D[level][i] << "\n";
			if (_D[level].size() < 100 || i % (_D[level].size() / 100))
				f.flush();
		}

	}



#endif


	return true;
}


template <typename T>
bool MultigridSolver<T>::prepareAtLevel(
	const T * D, ivec3 dim, Dir dir,
	uint level
)
{
	using vec3 = glm::tvec3<T, glm::highp>;
		
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);


	uint dirPrimary = getDirIndex(dir);
	uint dirSecondary[2] = { (dirPrimary + 1) % 3, (dirPrimary + 2) % 3 };
	const T highConc = 1.0f;
	const T lowConc = 0.0f;
	const T concetrationBegin = (getDirSgn(dir) == 1) ? highConc : lowConc;
	const T concetrationEnd = (getDirSgn(dir) == 1) ? lowConc : highConc;

	size_t N = dim.x * dim.y * dim.z;
	size_t M = N;

	


	
	//D getter
	const auto getD = [&dim,D](ivec3 pos) {			
		return D[linearIndex(dim, pos)];
	};

	const auto sample = [&getD, dim, D](ivec3 pos, Dir dir) {
		const int k = getDirIndex(dir);
		assert(pos.x >= 0 && pos.y >= 0 && pos.z >= 0);
		assert(pos.x < dim.x && pos.y < dim.y && pos.z < dim.z);

		ivec3 newPos = pos;
		int sgn = getDirSgn(dir);
		if (pos[k] + sgn < 0 || pos[k] + sgn >= dim[k]) {
			//do nothing
		}
		else {
			newPos[k] += getDirSgn(dir);
		}
		return getD(newPos);
	};	


	//Default spacing
	const vec3 h_general = { 1.0f / (dim.x + 1), 1.0f / (dim.y + 1), 1.0f / (dim.z + 1) };

	size_t d0Count = 0;
	size_t d1Count = 0;

		
	
	_f[level].resize(M);
	_x[level].resize(N);
	_A[level].resize(M, N);
	_A[level].reserve(Eigen::VectorXi::Constant(N, 7));
	


	//Build matrix
	for (auto z = 0; z < dim.z; z++) {
		for (auto y = 0; y < dim.y; y++) {
			for (auto x = 0; x < dim.x; x++) {

				const ivec3 ipos = { x,y,z };
				auto i = linearIndex(dim, ipos);

				vec3 hneg = h_general;
				vec3 hpos = h_general;

				//Adjust spacing on boundary
				for (auto k = 0; k < 3; k++) {
					if (ipos[k] == 0)
						hneg[k] *= 1.0 / sqrt(2);//0.5;					
					else if (ipos[k] >= dim[k] - 1)
						hpos[k] *= 1.0 / sqrt(2); //0.5;
				}
				const vec3 invHneg = vec3(1.0) / hneg;
				vec3 invHneg2 = invHneg*invHneg;
				const vec3 invHpos = vec3(1.0) / hpos;
				vec3 invHpos2 = invHpos*invHpos;

				T bval = 0.0f;
				T vals[7] = { 0.0f,0.0f ,0.0f ,0.0f ,0.0f ,0.0f ,0.0f };
				size_t colId[7];
				colId[0] = i - dim.y*dim.x;
				colId[1] = i - dim.x;
				colId[2] = i - 1;
				colId[3] = i;
				colId[4] = i + 1;
				colId[5] = i + dim.x;
				colId[6] = i + dim.y*dim.x;

				T Di = getD(ipos);
				


				auto Dvec = vec3(Di);

				auto Dneg = (vec3(
					sample(ipos, X_NEG),
					sample(ipos, Y_NEG),
					sample(ipos, Z_NEG)
				) + vec3(Dvec)) * T(0.5) * invHneg2;

				auto Dpos = (vec3(
					sample(ipos, X_POS),
					sample(ipos, Y_POS),
					sample(ipos, Z_POS)
				) + vec3(Dvec)) * T(0.5) * invHpos2;

				T diagVal = 0.0f;

				//(is first order accurate von neumann for secondary axes)
				for (auto k = 0; k < 3; k++) {
					if (ipos[k] > 0)
						diagVal += -Dneg[k];
					if (ipos[k] < dim[k] - 1)
						diagVal += -Dpos[k];
				}

				//Boundary conditions
				if (ipos[dirPrimary] == 0) {
					bval = -Dneg[dirPrimary] * concetrationBegin;
					diagVal -= Dneg[dirPrimary];

				}
				else if (ipos[dirPrimary] == dim[dirPrimary] - 1) {
					bval = -Dpos[dirPrimary] * concetrationEnd;
					diagVal -= Dpos[dirPrimary];
				}


				if (z > 0) vals[0] = Dneg.z;
				if (y > 0) vals[1] = Dneg.y;
				if (x > 0) vals[2] = Dneg.x;
				vals[3] = diagVal;
				if (x < dim.x - 1) vals[4] = Dpos.x;
				if (y < dim.y - 1) vals[5] = Dpos.y;
				if (z < dim.z - 1) vals[6] = Dpos.z;

				_f[level][i] = bval;

				//initial guess
				if (getDirSgn(dir) == 1)
					_x[level][i] = 1.0f - (ipos[dirPrimary] / T(dim[dirPrimary] + 1));
				else
					_x[level][i] = (ipos[dirPrimary] / T(dim[dirPrimary] + 1));


				for (auto k = 0; k < 7; k++) {
					if (colId[k] < 0 || colId[k] >= N || vals[k] == 0.0f) continue;
					_A[level].insert(i, colId[k]) = vals[k];
				}

			}
		}
	}


	//_porosity = T(d0Count) / T(d0Count + d1Count);

	_A[level].makeCompressed();

	//_solver.compute(_A);

	return true;
}






template <typename T>
T MultigridSolver<T>::solve(Volume &vol, T tolerance, size_t maxIterations)
{

	const int preN = 1;
	const int postN = 1;
	const int lastLevel = _lv - 1;

	Eigen::SparseLU<SparseMat> exactSolver;
	exactSolver.analyzePattern(_A[lastLevel]);
	exactSolver.factorize(_A[lastLevel]);
	
		
	auto tabs = [](int n){
		return std::string(n, '\t');
	};	

	

	if (_verbose) {
		addDebugChannel(vol, _x[0], _dims[0], "initial guess ", 0, true);
	}

	

	

	

	enum CycleType {
		V_CYCLE,
		W_CYCLE,
		V_CYCLE_SINGLE
	};

	CycleType ctype = V_CYCLE;
	std::vector<int> cycle; //+ down, -up

	if (ctype == V_CYCLE) {
		for (auto i = 0; i != _lv; i++) {
			cycle.push_back(i);
		}
		for (auto i = _lv - 2; i != -1; i--) {
			cycle.push_back(i);
		}
	}
	else if (ctype == W_CYCLE) {
		auto midLevel = (_lv-1) - 2;


		for (auto i = 0; i != _lv; i++) {
			cycle.push_back(i);
		}

		for (auto i = _lv - 2; i != (midLevel-1); i--) {
			cycle.push_back(i);
		}
		for (auto i = midLevel + 1; i != _lv ; i++) {
			cycle.push_back(i);
		}

		for (auto i = _lv - 2; i != -1; i--) {
			cycle.push_back(i);
		}

	}
	else if (ctype == V_CYCLE_SINGLE) {
		cycle = { 0, 1, 0 };
	}


	/////ZERO FIRST GUESS
	_x[0].setZero();

	Vector rInit = _f[0] - _A[0] * _x[0];

	T finalError = FLT_MAX;
	T lastError = sqrt(rInit.squaredNorm() / _f[0].squaredNorm());
	T lastResNorm = rInit.norm();
	T ratioSum = 0.0;
	//std::cout << "inital error: " << lastError << std::endl;	

	for (auto k = 0; k < maxIterations; k++) {

		int prevI = -1;
		for (auto i : cycle) {
			//Last level
			if (i == _lv - 1) {
				//Direct solver
				//if (_verbose) {
					//std::cout << tabs(lastLevel) << "Exact solve at Level " << lastLevel << std::endl;
				//}			
				
				//std::cout << "x pre exact " << i << ": " << v[i].squaredNorm() << std::endl;

				_x[lastLevel] = exactSolver.solve(_f[lastLevel]);

				//std::cout << "x post exact " << i << ": " << v[i].squaredNorm() << std::endl;
			}
			//Restrict
			else if (i > prevI) {
				//Initial guess is zero for all except first level
				if (i > 0) _x[i].setZero();

				
			//	std::cout << "x pre gs restr " << i << ": " << v[i].squaredNorm() << std::endl;

				//Pre smoother
				T err = solveGaussSeidel(_A[i], _f[i], _x[i], _r[i], _dims[i], tolerance, preN, false); // df = f  A*v
				//T err = solveJacobi(_A[i], _f[i], _x[i], _tmpx[i], _r[i], tolerance, preN, false); // df = f  A*v
				
				if (i == 0) {
					//std::cout << "V level " << i << " ||r||^2 = " << _r[i].squaredNorm() << std::endl;
				}
				

				//std::cout << "x post gs restr " << i << ": " << v[i].squaredNorm() << std::endl;
				
				//return 0;
				//addDebugChannel(vol, v[i], _dims[i], "v presmoothed", i, true);
				addDebugChannel(vol, _r[i], _dims[i], "r", i, true);

				//Restriction

				//std::cout << "f pre restr " << i + 1 << ": " << f[i + 1].squaredNorm() << std::endl;
				//restrictionWeighted(_dir, _r[i].data(), _dims[i], _f[i + 1].data(), _dims[i + 1], _D[i].data());
				//pointRestriction(_r[i].data(), _dims[i], _f[i + 1].data(), _dims[i + 1]);
				restriction(_r[i].data(), _dims[i], _f[i + 1].data(), _dims[i + 1]);
				addDebugChannel(vol, _f[i+1], _dims[i+1], "R(r)", i, true);
				
				//
				
				
				//std::cout << "f post restr " << i+1 << ": " << f[i+1].squaredNorm() << std::endl;
			
			}
			else {

				//std::cout << "x pre interp " << i + 1 << ": " << v[i + 1].squaredNorm() << std::endl;

				//Interpolation
				addDebugChannel(vol, _x[i+1], _dims[i+1], "x", i, true);
				//interpolationWeighted<T>(_x[i + 1].data(), _dims[i + 1], _tmpx[i].data(), _dims[i], _D[i + 1].data());
				interpolation<T>(_x[i + 1].data(), _dims[i + 1], _tmpx[i].data(), _dims[i]);
				//interp3D<T>(_x[i + 1].data(), _dims[i + 1], _tmpx[i].data(), _dims[i]);
				addDebugChannel(vol, _tmpx[i], _dims[i], "I(x)", i, true);
				//


				T coarse = _x[i + 1][linearIndex(_dims[i + 1], { 3,3,3 })];
				T fine = _tmpx[i][linearIndex(_dims[i], { 7,7,7 })];

				//std::cout << "xtemp post interp " << i << ": " << tmpx[i].squaredNorm() << std::endl;

				//addDebugChannel(vol, v[i], _dims[i], "v presmoothed", i, true);

				//Correction
				_x[i] += _tmpx[i];

				//Post smoothing, v[i] is initial guess & result		
				T err = solveGaussSeidel(_A[i], _f[i], _x[i], _r[i], _dims[i], tolerance, postN, false);
				//T err = solveJacobi(_A[i], _f[i], _x[i], _tmpx[i], _r[i], tolerance, preN, false); // df = f  A*v

				if (i == 0) {
					//std::cout << "^ level " << i << " ||r||^2 = " << _r[i].squaredNorm() << std::endl;
				}

				//addDebugChannel(vol, v[i], _dims[i], "v presmoothed", i, true);

			}

			prevI = i;
		}

		_iterations++;

		T resNorm = _r[0].norm();
		T err = sqrt(_r[0].squaredNorm() / _f[0].squaredNorm());

		

	/*	if (k % 10 == 0 && k > 0) {
			T maxR10 = _r[0].cwiseAbs().maxCoeff();
			T maxR0 = rInit.cwiseAbs().maxCoeff();
			T rho = pow(maxR10 / maxR0, 1.0 / 10.0);
			rInit = _r[0];
			std::cout << "k = " << k << " err: " << err << ", rho: " << rho << std::endl;
		}*/


		std::cout << "k = " << k << " | |r|: " << resNorm << ", ratio: " << resNorm / lastResNorm;
		std::cout << " | |e|: " << err << ", ratio: " << err / lastError << std::endl;

		ratioSum += resNorm / lastResNorm;

		lastResNorm = resNorm;
		lastError = err;

		if (err < tolerance || isinf(err) || isnan(err)) {
			finalError = err;
			break;
		}
			


	}
	std::cout << "Avg ratio: " << ratioSum / _iterations << std::endl;



	//Point test	
	/*{
	
		_iterations == 0;

		auto i = 0;
		T err = FLT_MAX;
		for (auto k = 0; k < 100; k++) {
			 T newErr = solveGaussSeidel(_A[i], _f[i], _x[i], _r[i], _dims[i], 1e-24, 128, true);
			 if (err == newErr)
				 break;
			 err = newErr;
		}	

		std::cout << "exact solution reached ? err  = " << err << std::endl;

		for (auto k = 0; k < 1; k++){
			int prevI = -1;
			for (auto i : cycle) {
				//Last level
				if (i == _lv - 1) {
					std::cout << "========= Level " << i << " EXACT" << std::endl;
					_x[lastLevel] = exactSolver.solve(_f[lastLevel]);				
				}				
				else if (i > prevI) {				
					if (i > 0) _x[i].setZero();
					std::cout << "========= Level " << i << " V" << std::endl;					
					T err = solveGaussSeidel(_A[i], _f[i], _x[i], _r[i], _dims[i], tolerance, preN, false); 
					std::cout << "presmooth err = " << err << std::endl;
					std::cout << "presmooth residual sqnorm = " << _r[i].squaredNorm() << std::endl;
					std::cout << "presmooth b sqnorm = " << _f[i].squaredNorm() << std::endl;
					restrictionWeighted(_dir, _r[i].data(), _dims[i], _f[i + 1].data(), _dims[i + 1], _D[i].data());
					std::cout << "restricted resiudual sqnorm = " << _f[i + 1].squaredNorm() << std::endl;
				}
				else {	
					std::cout << "========= Level " << i << " ^" << std::endl;
					interpolationWeighted<T>(_x[i + 1].data(), _dims[i + 1], _tmpx[i].data(), _dims[i], _D[i + 1].data());
					std::cout << "correction sq norm: " << _tmpx[i].squaredNorm() << std::endl;
					_x[i] += _tmpx[i];	
					T err = solveGaussSeidel(_A[i], _f[i], _x[i], _r[i], _dims[i], tolerance, postN, false);					
				}

				prevI = i;
			}

			_iterations++;
			T err = sqrt(_r[0].squaredNorm() / _f[0].squaredNorm());
			lastError = err;
						
			
			std::cout << "k = " << k << " err: " << err << std::endl;
		}
	}*/



	return finalError;

}


template <typename T>
BLIB_EXPORT bool blib::MultigridSolver<T>::resultToVolume(VolumeChannel & vol)
{

	assert(_x.size() > 0);

	void * destPtr = vol.getCurrentPtr().getCPU();

	//Copy directly if same type
	if ((std::is_same<float, T>::value && vol.type() == TYPE_FLOAT) ||
		(std::is_same<double, T>::value && vol.type() == TYPE_DOUBLE)) {
		memcpy(destPtr, _x[0].data(), _x[0].size() * sizeof(T));
	}
	else {
		if (vol.type() == TYPE_FLOAT) {
			Eigen::Matrix<float, Eigen::Dynamic, 1> tmpX = _x[0].cast<float>();
			memcpy(destPtr, tmpX.data(), tmpX.size() * sizeof(float));
		}
		else if (vol.type() == TYPE_DOUBLE) {
			Eigen::Matrix<double, Eigen::Dynamic, 1> tmpX = _x[0].cast<double>();
			memcpy(destPtr, tmpX.data(), tmpX.size() * sizeof(double));
		}
		else {
			return false;
		}
	}

	return true;
}





template <typename T>
BLIB_EXPORT bool blib::MultigridSolver<T>::generateErrorVolume(Volume & v)
{

	//assert(std::is_same<float, T>::value == true);

	for (uint i = 0; i < _lv; i++) {		
		auto id = v.addChannel(_dims[i], TYPE_FLOAT);
		auto & c = v.getChannel(id);
		memcpy(c.getCurrentPtr().getCPU(), _r[i].data(), _dims[i].x * _dims[i].y * _dims[i].z * sizeof(float));
		c.getCurrentPtr().commit();				
	}

	return true;
}



template <typename T>
BLIB_EXPORT T blib::MultigridSolver<T>::tortuosity(const VolumeChannel & mask, Dir dir)
{
	assert(mask.type() == TYPE_UCHAR || mask.type() == TYPE_CHAR);
	const auto dim = mask.dim();

	const T * concData = _x[0].data();
	const uchar * cdata = (uchar *)mask.getCurrentPtr().getCPU();

	const int primaryDim = getDirIndex(dir);
	const int secondaryDims[2] = { (primaryDim + 1) % 3, (primaryDim + 2) % 3 };


	int n = dim[secondaryDims[0]] * dim[secondaryDims[1]];
	int k = (getDirSgn(dir) == -1) ? 0 : dim[primaryDim] - 1;

	/*
	Calculate average in low concetration plane
	*/
	bool zeroOutPart = true;
	std::vector<T> isums(dim[secondaryDims[0]], T(0));

	#pragma omp parallel for
	for (auto i = 0; i < dim[secondaryDims[0]]; i++) {
		T jsum = T(0);
		for (auto j = 0; j < dim[secondaryDims[1]]; j++) {
			ivec3 pos;
			pos[primaryDim] = k;
			pos[secondaryDims[0]] = i;
			pos[secondaryDims[1]] = j;

			if (zeroOutPart && cdata[linearIndex(dim, pos)] == 0)
				jsum += concData[linearIndex(dim, pos)];
		}
		isums[i] = jsum;
	}

	T sum = std::accumulate(isums.begin(), isums.end(), T(0));

	double dc = sum / n;
	double dx = 1.0f / (dim[primaryDim] + 1);
	double tau = /*dx * */_porosity / (dc * /*dx **/ dim[primaryDim] * 2);

	if (_verbose) {
		std::cout << "dc: " << dc << std::endl;
		std::cout << "porosity: " << _porosity << std::endl;
		std::cout << "tau: " << tau << std::endl;
	}

	return tau;
}
