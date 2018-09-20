#pragma once

#include <cuda_runtime.h>

//https://stackoverflow.com/questions/12778949/cuda-memory-alignment
#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif


#define VOLUME_VOX					\
uint3 vox = make_uint3(			\
		blockIdx.x * blockDim.x,	\
		blockIdx.y * blockDim.y,	\
		blockIdx.z * blockDim.z		\
	) + threadIdx;					\

#define VOLUME_IVOX					\
int3 ivox = make_int3(			\
		blockIdx.x * blockDim.x,	\
		blockIdx.y * blockDim.y,	\
		blockIdx.z * blockDim.z		\
	) + make_int3(threadIdx);					\

#define VOLUME_BLOCK_IVOX					\
int3 blockIvox = make_int3(			\
		blockIdx.x * blockDim.x,	\
		blockIdx.y * blockDim.y,	\
		blockIdx.z * blockDim.z		\
	);					\


#define VOLUME_VOX_GUARD(res)					\
	VOLUME_VOX									\
	if (vox.x >= res.x || vox.y >= res.y || vox.z >= res.z)	\
	return;		

#define VOLUME_IVOX_GUARD(res)					\
	VOLUME_IVOX									\
	if (ivox.x >= res.x || ivox.y >= res.y || ivox.z >= res.z)	\
	return;		

#define BLOCKS3D(perBlockDim, res)					\
	if(perBlockDim*perBlockDim*perBlockDim > 1024){	\
		printf("Block too big %d > %d", perBlockDim*perBlockDim*perBlockDim, 1024);	\
		exit(1);								\
	}											\
	uint3 block = make_uint3(perBlockDim);		\
	uint3 numBlocks = make_uint3(				\
		((res.x + (block.x-1)) / block.x),		\
		((res.y + (block.y-1)) / block.y),		\
		((res.z + (block.z-1)) / block.z)		\
	);											\



#define BLOCKS3D_INT3(a,b,c, res)					\
	if(a*b*c > 1024){	\
		printf("Block too big %d > %d", a*b*c, 1024);	\
		exit(1);								\
	}											\
	uint3 block = make_uint3(a,b,c);		\
	uint3 numBlocks = make_uint3(				\
		((res.x + (block.x-1)) / block.x),		\
		((res.y + (block.y-1)) / block.y),		\
		((res.z + (block.z-1)) / block.z)		\
	);

								\

__host__ __device__ inline uint roundDiv(uint a, uint b) {
	return (a + (b - 1)) / b;
}

__host__ __device__ inline uint3 roundDiv(uint3 a, uint3 b) {
	return make_uint3(
		(a.x + (b.x - 1)) / b.x,
		(a.y + (b.y - 1)) / b.y,
		(a.z + (b.z - 1)) / b.z
	);
}

__host__ __device__ inline int3 dirVec(Dir d) {
	switch (d) {
	case X_POS: return make_int3(1, 0, 0);
	case X_NEG: return make_int3(-1, 0, 0);
	case Y_POS: return make_int3(0, 1, 0);
	case Y_NEG: return make_int3(0, -1, 0);
	case Z_POS: return make_int3(0, 0, 1);
	case Z_NEG: return make_int3(0, 0, -1);
	};
	return make_int3(0, 0, 0);
}


inline __device__ uint _getDirIndex(Dir dir) {
	switch (dir) {
	case X_POS:
	case X_NEG:
		return 0;
	case Y_POS:
	case Y_NEG:
		return 1;
	case Z_POS:
	case Z_NEG:
		return 2;
	}
	return uint(-1);
}


template<typename D, typename P>
inline __device__ __host__ size_t _linearIndex(const D & dim, const P & pos) {
	return pos.x + dim.x * pos.y + dim.x * dim.y * pos.z;
}

inline __device__ __host__ size_t _linearIndexXFirst(const uint3 & dim, const int3 & pos) {
	return pos.z + dim.z * pos.y + dim.z * dim.y * pos.x;
}

inline __device__ __host__ bool _isValidPos(const uint3 & dim, const uint3 & pos) {
	return pos.x < dim.x && pos.y < dim.y && pos.z < dim.z;
}

inline __device__ __host__ bool _isValidPos(const uint3 & dim, const int3 & pos) {
	return	pos.x >= 0 && pos.y >= 0 && pos.z >= 0 &&
		pos.x < int(dim.x) && pos.y < int(dim.y) && pos.z < int(dim.z);
}

inline __device__ __host__ int3 posFromLinear(const int3 & dim, int index) {

	int3 pos;
	pos.x = (index % dim.x);
	index = (index - pos.x) / dim.x;
	pos.y = (index % dim.y);
	pos.z = (index / dim.y);
	return pos;
}


inline __device__ __host__ int _getDirSgn(Dir dir) {
	return -((dir % 2) * 2 - 1);
}

inline __device__ __host__ Dir _getDir(int index, int sgn) {
	sgn = (sgn + 1) / 2; // 0 neg, 1 pos
	sgn = 1 - sgn; // 1 neg, 0 pos
	return Dir(index * 2 + sgn);
}

template <typename T, typename VecType>
inline __device__ __host__ T & _at(VecType & vec, int index) {
	return ((T*)&vec)[index];
}
template <typename T, typename VecType>
inline __device__ __host__ const T & _at(const VecType & vec, int index) {
	return ((T*)&vec)[index];
}


inline __device__ uint3 clampedVox(const uint3 & res, uint3 vox, Dir dir) {
	const int k = _getDirIndex(dir);
	int sgn = _getDirSgn(dir); //todo better

	const int newVox = _at<int>(vox, k) + sgn;
	const int & resK = _at<int>(res, k);
	if (newVox >= 0 && newVox < resK) {
		_at<int>(vox, k) = uint(newVox);
	}
	return vox;
}

inline __device__ uint3 periodicVox(const uint3 & res, uint3 vox, Dir dir) {
	const int k = _getDirIndex(dir);
	int sgn = _getDirSgn(dir); 	
	const int & resK = _at<int>(res, k);
	const int newVox = (_at<int>(vox, k) + sgn + resK) % resK;	
	_at<uint>(vox, k) = uint(newVox);	
	return vox;
}

/*
Templated surface write
*/
template <typename T>
inline __device__ void write(cudaSurfaceObject_t surf, const uint3 & vox, const T & val);

template <typename T>
inline __device__ void write(cudaSurfaceObject_t surf, const int3 & vox, const T & val);


template<>
inline __device__ void write(cudaSurfaceObject_t surf, const uint3 & vox, const float & val) {
#ifdef __CUDA_ARCH__
	surf3Dwrite(val, surf, vox.x * sizeof(float), vox.y, vox.z);
#endif
}

template<>
inline __device__ void write(cudaSurfaceObject_t surf, const uint3 & vox, const double & val) {
	
#ifdef __CUDA_ARCH__
	const int2 * valInt = (int2*)&val;
	surf3Dwrite(*valInt, surf, vox.x * sizeof(int2), vox.y, vox.z);
#endif
}

template<>
inline __device__ void write(cudaSurfaceObject_t surf, const int3 & vox, const double & val) {

#ifdef __CUDA_ARCH__
	const int2 * valInt = (int2*)&val;
	surf3Dwrite(*valInt, surf, vox.x * sizeof(int2), vox.y, vox.z);
#endif
}


/*
Templated surface read (direct)
*/
template <typename T>
inline __device__ T read(cudaSurfaceObject_t surf, const uint3 & vox);

template <typename T>
inline __device__ T read(cudaSurfaceObject_t surf, const int3 & vox);

template<>
inline __device__ float read(cudaSurfaceObject_t surf, const uint3 & vox) {
	float val = 0.0f;
#ifdef __CUDA_ARCH__
	surf3Dread(&val, surf, vox.x * sizeof(float), vox.y, vox.z);
#endif
	return val;
}

template<>
inline __device__ uchar read(cudaSurfaceObject_t surf, const uint3 & vox) {
	uchar val = 0;
#ifdef __CUDA_ARCH__
	surf3Dread(&val, surf, vox.x * sizeof(uchar), vox.y, vox.z);
#endif
	return val;
}

template<>
inline __device__ double read(cudaSurfaceObject_t surf, const uint3 & vox) {
	
#ifdef __CUDA_ARCH__
	int2 val;
	surf3Dread(&val, surf, vox.x * sizeof(int2), vox.y, vox.z);
	return __hiloint2double(val.y, val.x);
#else
	return 0.0;
#endif
}

template<>
inline __device__ double read(cudaSurfaceObject_t surf, const int3 & vox) {

#ifdef __CUDA_ARCH__
	int2 val;
	surf3Dread(&val, surf, vox.x * sizeof(int2), vox.y, vox.z);
	return __hiloint2double(val.y, val.x);
#else
	return 0.0;
#endif
}


struct CUDA_Volume {	
	uint3 res;
	PrimitiveType type;
	cudaSurfaceObject_t surf;

	//ID given by Host code
	int ID;
};

template <size_t T, size_t size>
struct CUDA_Kernel3D {
	double v[size][size][size];
};

/*template <size_t size>
using CUDA_Kernel3Dd = CUDA_Kernel3D<double, size>;

template <size_t size>
using CUDA_Kernel3Df = CUDA_Kernel3D<float, size>;

template <typename T>
using CUDA_KernelPtr = T *;

using CUDA_KernelPtrD = CUDA_KernelPtr<double>;
using CUDA_KernelPtrF = CUDA_KernelPtr<float>;
*/