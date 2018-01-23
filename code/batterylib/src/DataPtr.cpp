#include "DataPtr.h"

#include "CudaUtility.h"
#include "GLGlobal.h"

#include <assert.h>
#include <cuda_gl_interop.h>

bool blib::DataPtr::retrieve(uint offset, uint size)
{
	assert(cpu != nullptr);
	assert(gpu != nullptr);
	assert(stride > 0);
	assert(size > 0);

	return _CUDA(
		cudaMemcpy((uchar*)cpu + stride * offset, (uchar*)gpu + stride * offset, size, cudaMemcpyDeviceToHost)
	);
}

bool blib::DataPtr::retrieve()
{
	return retrieve(0, byteSize());
}

bool blib::DataPtr::allocHost(uint num, uint stride)
{
	if (cpu) {
		delete[] cpu;
		cpu = nullptr;
	}

	cpu = new uchar[num*stride];
	return true;

}

bool blib::DataPtr::allocDevice(uint num, uint stride)
{	
	if (gpu) {
		_CUDA(cudaFree(gpu));
		gpu = nullptr;
	}

	if (_CUDA(cudaMalloc(&gpu, num*stride))){
		this->stride = stride;
		this->num = num;
		return true;
	}

	return false;
}


blib::DataPtr::~DataPtr()
{
	if (cpu) delete[] cpu;
	if(gpu) _CUDA(cudaFree(gpu));	
}

bool blib::DataPtr::commit(uint offset, uint size) {

	assert(cpu != nullptr);
	assert(gpu != nullptr);
	assert(stride > 0);
	assert(size > 0);

	return _CUDA(
		cudaMemcpy((char*)gpu + stride * offset, (char*)cpu + stride * offset, size, cudaMemcpyHostToDevice)
	);
}

bool blib::DataPtr::commit()
{
	return commit(0, byteSize());
}

//////////////////


blib::Texture3DPtr::Texture3DPtr()
{
	_cpu.ptr = nullptr;
	_cpu.pitch = 0;
	_cpu.xsize = 0;
	_cpu.ysize = 0;

	_gpu = nullptr;

	_desc.x = 0;
	_desc.y = 0;
	_desc.z = 0;
	_desc.w = 0;
	_desc.f = cudaChannelFormatKindUnsigned;

	_extent.width = 0;
	_extent.height = 0;
	_extent.depth = 0;

	_glID = 0;
}


bool blib::Texture3DPtr::alloc(PrimitiveType type, ivec3 dim, bool alsoOnCPU)
{
	
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);

	_extent = make_cudaExtent(dim.x, dim.y, dim.z);

	setDesc(type);

	if (!_CUDA(cudaMalloc3DArray(&_gpu, &_desc, _extent, cudaArraySurfaceLoadStore)))
		return false;

	
	if (!alsoOnCPU)
		return true;
	
	void * hostPtr = new char[byteSize()];
	_cpu = make_cudaPitchedPtr(hostPtr, stride() * _extent.width, _extent.width, _extent.height);

	return true;

}

bool blib::Texture3DPtr::allocOpenGL(PrimitiveType type, ivec3 dim, bool alsoOnCPU /*= false*/)
{
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);
	
	//
	assert(_glID == 0); //no realloc yet
	assert(_gpu == nullptr);
	//


	_extent = make_cudaExtent(dim.x, dim.y, dim.z);
	setDesc(type);
	
	
	if (alsoOnCPU) {
		auto size = byteSize();
		void * hostPtr = new char[size];
		_cpu = make_cudaPitchedPtr(hostPtr, stride() * _extent.width, _extent.width, _extent.height);
	}		
		
	GL(glGenTextures(1, (GLuint*)&_glID));
	GL(glBindTexture(GL_TEXTURE_3D, _glID));
	GL(glPixelStorei(GL_PACK_ALIGNMENT, 4));
	GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 4));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
	
	switch (type) {
		case TYPE_UCHAR:
		case TYPE_CHAR:	
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, dim.x, dim.y, dim.z, 0, GL_RED, GL_UNSIGNED_BYTE, 0));	
			break;	
		case TYPE_FLOAT:	
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, dim.x, dim.y, dim.z, 0, GL_RED, GL_FLOAT, 0));	
			break;
		default:
			assert("Not implemented");
			break;
	};
	

	//Register texture for interop
	GL(_CUDA(cudaGraphicsGLRegisterImage(&_gpuRes, _glID, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore)));

	//Map resource for cuda
	_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));

	//Get pointer
	_CUDA(cudaGraphicsSubResourceGetMappedArray(&_gpu, _gpuRes, 0, 0));
	
	//Create surface
	createSurface();

	//Unmap resource
	_CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));	

	
	
	return true;
}

bool blib::Texture3DPtr::commit()
{

	assert(_cpu.ptr != nullptr);
	assert(_gpu != nullptr);

	cudaMemcpy3DParms p;
	memset(&p, 0, sizeof(cudaMemcpy3DParms));

	p.extent = _extent;
	p.kind = cudaMemcpyHostToDevice;
	
	p.srcPtr = _cpu;	
	p.dstArray = _gpu;
	
	_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));
	bool res = _CUDA(cudaMemcpy3D(&p));
	_CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));
	return res;


	/*
	//Alternatively use glTexImage3D
		GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, _extent.width, _extent.height, _extent.depth, 0, GL_RED, GL_FLOAT, _cpu.ptr));
		return true;
	*/
}


bool blib::Texture3DPtr::retrieve()
{

	assert(_cpu.ptr != nullptr);
	assert(_gpu != nullptr);

	cudaMemcpy3DParms p;
	memset(&p, 0, sizeof(cudaMemcpy3DParms));

	p.extent = _extent;
	p.kind = cudaMemcpyDeviceToHost;

	p.srcArray = _gpu;
	p.dstPtr = _cpu;	

	_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));
	bool res = _CUDA(cudaMemcpy3D(&p));
	_CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));

	return res;
}

bool blib::Texture3DPtr::copySurfaceTo(void * gpuSurfacePtr) const
{
	return _CUDA(
		cudaMemcpy(gpuSurfacePtr, &_surface, sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice)
	);
}

bool blib::Texture3DPtr::createSurface()
{
	cudaResourceDesc resDesc; 

	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = _gpu;
	
	return _CUDA(cudaCreateSurfaceObject(&_surface, &resDesc));
}

void blib::Texture3DPtr::setDesc(PrimitiveType type)
{
	switch (type) {

	case TYPE_FLOAT3:
		_desc.x = sizeof(float);
		_desc.y = sizeof(float);
		_desc.z = sizeof(float);
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindFloat;
		break;
	case TYPE_FLOAT4:
		_desc.x = sizeof(float);
		_desc.y = sizeof(float);
		_desc.z = sizeof(float);
		_desc.w = sizeof(float);
		_desc.f = cudaChannelFormatKindFloat;
		break;
	case TYPE_FLOAT:
		_desc.x = sizeof(float);
		_desc.y = 0;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindFloat;
		break;
	case TYPE_UCHAR:
		//
	default:
		_desc.x = sizeof(char);
		_desc.y = 0;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindUnsigned;
		break;
	};

}
