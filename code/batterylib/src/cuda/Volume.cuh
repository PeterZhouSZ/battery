#include <cuda_runtime.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>

#include "CudaMath.h"
#include "PrimitiveTypes.h"


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
	//printf("Launching kernel with %d,%d,%d blocks, per block: %d,%d,%d\n", numBlocks.x, numBlocks.y,numBlocks.z, block.x, block.y, block.z)\
	

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
	);											\

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

inline __device__ __host__ size_t _linearIndex(const uint3 & dim, const uint3 & pos) {
	return pos.x + dim.x * pos.y + dim.x * dim.y * pos.z;
}

inline __device__ __host__ size_t _linearIndex(const uint3 & dim, const int3 & pos) {
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




void launchErodeKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut);

void launchHeatKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut);

void launchBinarizeKernel(uint3 res, 
	cudaSurfaceObject_t surfInOut, 
	PrimitiveType type,
	float threshold //between 0 to 1
);

#define BOUNDARY_ZERO_GRADIENT 1e37f
struct DiffuseParams {
	uint3 res;
	float voxelSize; 
	cudaSurfaceObject_t mask;
	cudaSurfaceObject_t concetrationIn;
	cudaSurfaceObject_t concetrationOut;
	float zeroDiff;
	float oneDiff;
		
	float boundaryValues[6];
};

void launchDiffuseKernel(DiffuseParams params);



//Subtracts B from A (result in A) ... A = A -B
void launchSubtractKernel(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B);

//Inplace reduce sum
float launchReduceSumKernel(uint3 res, cudaSurfaceObject_t surf);


float launchReduceSumSlice(uint3 res, cudaSurfaceObject_t surf, Dir dir, void * output);

enum ReduceOpType {
	REDUCE_OP_SUM,
	REDUCE_OP_PROD,
	REDUCE_OP_SQUARESUM,
	REDUCE_OP_MIN,
	REDUCE_OP_MAX
};

#define VOLUME_REDUCTION_BLOCKSIZE 512
void launchReduceKernel(
	PrimitiveType type,
	ReduceOpType opType,
	uint3 res,
	cudaSurfaceObject_t surf,
	void * auxBufferGPU,
	void * auxBufferCPU,
	void * result
);



void launchClearKernel(
	PrimitiveType type, cudaSurfaceObject_t surf, uint3 res, void * val
);