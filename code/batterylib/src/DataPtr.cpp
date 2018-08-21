#include "DataPtr.h"

#include "CudaUtility.h"
#include "GLGlobal.h"

#include <assert.h>
#include <cuda_gl_interop.h>


blib::DataPtr::DataPtr()
{
	memset(this, 0, sizeof(DataPtr));	
}

blib::DataPtr & blib::DataPtr::operator=(blib::DataPtr &&other)
{
	memcpy(this, &other, sizeof(other));
	memset(&other, 0, sizeof(DataPtr));
	return *this;
}

blib::DataPtr::DataPtr(blib::DataPtr &&other)
{
	memcpy(this, &other, sizeof(other));
	memset(&other, 0, sizeof(DataPtr));
}

bool blib::DataPtr::retrieve(size_t offset, size_t size)
{
	
	assert(gpu != nullptr);
	assert(stride > 0);
	assert(size > 0);

	if (!cpu) {
		allocHost(num, stride);
	}

	return _CUDA(
		cudaMemcpy((uchar*)cpu + stride * offset, (uchar*)gpu + stride * offset, size, cudaMemcpyDeviceToHost)
	);
}

bool blib::DataPtr::retrieve()
{
	return retrieve(0, byteSize());
}

bool blib::DataPtr::allocHost(size_t num, size_t stride)
{
	if (cpu) {

		if (num == this->num && stride == this->stride)
			return true;

		delete[] cpu;
		cpu = nullptr;
	}

	cpu = new uchar[num*stride];
	return true;

}

bool blib::DataPtr::allocDevice(size_t num, size_t stride)
{	
	if (gpu) {
		if (num == this->num && stride == this->stride) 
			return true;

		
		_CUDA(cudaFree(gpu));
		gpu = nullptr;
	}

	if (_CUDA(cudaMalloc((void **)&gpu, num*stride)) && gpu != nullptr){
		this->stride = stride;
		this->num = num;
		return true;
	}
	else {
		assert(false);		
	}

	return false;
}


bool blib::DataPtr::alloc(size_t num, size_t stride)
{
	return allocDevice(num, stride) && allocHost(num, stride);
}

blib::DataPtr::~DataPtr()
{
	if (cpu) {
		delete[] cpu;
		cpu = nullptr;
	}
	if (gpu) {
		_CUDA(cudaFree(gpu));
		gpu = nullptr;
	}
}

bool blib::DataPtr::commit(size_t offset, size_t size) {

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
	memset(this, 0, sizeof(Texture3DPtr));
	_desc.f = cudaChannelFormatKindUnsigned;	
}



blib::Texture3DPtr & blib::Texture3DPtr::operator=(blib::Texture3DPtr &&other)
{
	memcpy(this, &other, sizeof(other));	
	memset(&other, 0, sizeof(Texture3DPtr));
	return *this;
}

blib::Texture3DPtr::Texture3DPtr(blib::Texture3DPtr &&other)
{
	memcpy(this, &other, sizeof(other));
	memset(&other, 0, sizeof(Texture3DPtr));
}

blib::Texture3DPtr::~Texture3DPtr()
{
	if (_cpu.ptr) {
		delete[] _cpu.ptr;		
	}

	if (_gpu) {		
		_CUDA(cudaDestroySurfaceObject(_surface));	
		if (_glID != 0) {
			_CUDA(cudaGraphicsUnregisterResource(_gpuRes));			
			GL(glBindTexture(GL_TEXTURE_3D, _glID));			
			GL(glDeleteTextures(1, &_glID));
		}		
	}

	memset(this, 0, sizeof(Texture3DPtr));
}

bool blib::Texture3DPtr::alloc(PrimitiveType type, ivec3 dim, bool alsoOnCPU)
{
	
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);

	_extent = make_cudaExtent(dim.x, dim.y, dim.z);

	setDesc(type);

	if (!_CUDA(cudaMalloc3DArray(&_gpu, &_desc, _extent, 0)))
		return false;

	
	createSurface();

	if (!alsoOnCPU)
		return true;
	
	void * hostPtr = new char[byteSize()];
	_cpu = make_cudaPitchedPtr(hostPtr, stride() * _extent.width, _extent.width, _extent.height);

	return true;

}

bool blib::Texture3DPtr::allocOpenGL(PrimitiveType type, ivec3 dim, bool alsoOnCPU /*= false*/)
{
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);
	
	

	if (_gpu != nullptr && _type == type) {
		auto newExtent = make_cudaExtent(dim.x, dim.y, dim.z);
		if (_extent.depth == newExtent.depth && _extent.height == newExtent.height && _extent.width == newExtent.width)
			return true;		
	}

	//
	//assert(_glID == 0); //no realloc yet
	//assert(_gpu == nullptr);
	//


	_extent = make_cudaExtent(dim.x, dim.y, dim.z);
	
	//Set channel descriptor (elements, sizes)
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


	const bool linear = true;
	if (linear) {
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	}
	else {
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	}
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
	
	switch (type) {
		case TYPE_UCHAR:
		case TYPE_CHAR:	
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, dim.x, dim.y, dim.z, 0, GL_RED, GL_UNSIGNED_BYTE, NULL));	
			break;	
		case TYPE_FLOAT:	
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, dim.x, dim.y, dim.z, 0, GL_RED, GL_FLOAT, NULL));
			break;
		case TYPE_DOUBLE:
			//Double ~ reinterpreted int2
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32I, dim.x, dim.y, dim.z, 0, GL_RG_INTEGER, GL_UNSIGNED_INT, NULL));
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

bool blib::Texture3DPtr::mapGPUArray()
{
	//Map resource for cuda
	_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));

	//Get pointer
	return _CUDA(cudaGraphicsSubResourceGetMappedArray(&_gpu, _gpuRes, 0, 0));
}

 bool blib::Texture3DPtr::unmapGPUArray()
{
	 //Unmap resource
	 return _CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));
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
	
	if(_glID != 0)
		_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));

	bool res = _CUDA(cudaMemcpy3D(&p));
	
	if (_glID != 0)
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

	if (_glID != 0)
		_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));
	bool res = _CUDA(cudaMemcpy3D(&p));

	if (_glID != 0)
		_CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));

	return res;
}

bool blib::Texture3DPtr::copySurfaceTo(void * gpuSurfacePtr) const
{
	return _CUDA(
		cudaMemcpy(gpuSurfacePtr, &_surface, sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice)
	);
}

bool blib::Texture3DPtr::copyTo(DataPtr & ptr)
{
	assert(ptr.byteSize() == byteSize());
	assert(ptr.stride == stride());
	bool res = true;
	res &= mapGPUArray();

	cudaMemcpy3DParms p;
	memset(&p, 0, sizeof(cudaMemcpy3DParms));

	p.extent = _extent;
	p.kind = cudaMemcpyDeviceToDevice;

	p.srcArray = _gpu;

	auto gpuPitched = make_cudaPitchedPtr(
		ptr.gpu, stride() * _extent.width, _extent.width, _extent.height
	);

	p.dstPtr = gpuPitched;

	res &= _CUDA(cudaMemcpy3D(&p));

	res &= unmapGPUArray();
	return res;
}

bool blib::Texture3DPtr::copyFrom(DataPtr & ptr)
{
	assert(ptr.byteSize() == byteSize());
	assert(ptr.stride == stride());
	bool res = true;
	res &= mapGPUArray();

	cudaMemcpy3DParms p;
	memset(&p, 0, sizeof(cudaMemcpy3DParms));

	p.extent = _extent;
	p.kind = cudaMemcpyDeviceToDevice;

	auto gpuPitched = make_cudaPitchedPtr(
		ptr.gpu, stride() * _extent.width, _extent.width, _extent.height
	);

	p.srcPtr = gpuPitched;

	p.dstArray = _gpu;

	res &= _CUDA(cudaMemcpy3D(&p));

	res &= unmapGPUArray();
	return res;
}

bool blib::Texture3DPtr::clear(uchar val /*= 0*/)
{
	memset(getCPU(), val, this->byteSize());	
	return commit();

}

bool blib::Texture3DPtr::clearGPU(uchar val /*= 0*/)
{
	/*_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));
	cudaPitchedPtr ptr;
	ptr.ptr = _gpu;
	ptr.pitch = _extent.width * stride();
	ptr.xsize = _extent.width * stride();
	ptr.ysize = _extent.width * _extent.height * stride();
	bool res = _CUDA(cudaMemset3D(ptr, val, _extent));
	_CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));*/
	return false;
	//return res;
}

bool blib::Texture3DPtr::fillSlow(void * elem)
{
	auto * cpu = getCPU();
	
	const auto numElem = num();
	const auto size = primitiveSizeof(_type);
	for (auto i = 0; i < numElem; i++){
		memcpy((char*)cpu + i*size, elem, size);
	}
	
	return commit();

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
		_desc.x = sizeof(float) * 8;
		_desc.y = sizeof(float) * 8;
		_desc.z = sizeof(float) * 8;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindFloat;
		break;
	case TYPE_FLOAT4:
		_desc.x = sizeof(float) * 8;
		_desc.y = sizeof(float) * 8;
		_desc.z = sizeof(float) * 8;
		_desc.w = sizeof(float) * 8;
		_desc.f = cudaChannelFormatKindFloat;
		break;
	case TYPE_FLOAT:
		_desc.x = sizeof(float) * 8;
		_desc.y = 0;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindFloat;
		break;
	case TYPE_DOUBLE:
		_desc.x = (sizeof(int2) / 2) * 8;
		_desc.y = (sizeof(int2) / 2) * 8;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindSigned;
		break;
	case TYPE_UCHAR:
		//
	default:
		_desc.x = sizeof(unsigned char) * 8;
		_desc.y = 0;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindUnsigned;
		break;
	};

	_type = type;

}
