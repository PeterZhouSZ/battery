#include "Volume.h"
#include "GLGlobal.h"

#include "Volume.cuh"
#include "CudaUtility.h"

using namespace blib;

bool glewInited = false;

blib::VolumeChannel::VolumeChannel(ivec3 dim, PrimitiveType type, bool doubleBuffered)
	: dim(dim), type(type), _current(0), _doubleBuffered(doubleBuffered)
{	


	//Allocate buffer(s)
	_ptr[_current].allocOpenGL(type, dim, true);
	if (doubleBuffered) {
		_ptr[(_current + 1) % 2].allocOpenGL(type, dim, true);	
	}

	
	//Test fill
	auto & ptr = getCurrentPtr();	
	float * arr = reinterpret_cast<float*>(ptr.getCPU());	
	memset(arr, 0, ptr.byteSize());

	/*srand(0);
	for (auto i = 0; i < ptr.byteSize() / ptr.stride(); i++) {
		/ *float val = (rand() % 255) / 255.0f;
		if (val < 0.1f) val = 0.0f;
		if (val > 0.60f) val = 1.0f;

		((float*)arr)[i] = val;* /

		if (i < ptr.byteSize() / ptr.stride() / 2)
			((float*)arr)[i] = 1.0f;
		else
			((float*)arr)[i] = 0.0f;

	}	*/
	ptr.commit();
}


blib::Texture3DPtr & blib::VolumeChannel::getCurrentPtr()
{
	return _ptr[_current];
}

const Texture3DPtr & blib::VolumeChannel::getNextPtr() const
{
	//Attempting to get double buffer when channel was allocated as single buffer
	assert(_doubleBuffered);
	return _ptr[(_current + 1) % 2];
}

Texture3DPtr & blib::VolumeChannel::getNextPtr()
{
	//Attempting to get double buffer when channel was allocated as single buffer
	assert(_doubleBuffered); 
	return _ptr[(_current + 1) % 2];
}

const blib::Texture3DPtr & blib::VolumeChannel::getCurrentPtr() const
{
	return _ptr[_current];
}

void blib::VolumeChannel::swapBuffers()
{
	//Attempting to get double buffer when channel was allocated as single buffer
	assert(_doubleBuffered);
	if (_doubleBuffered) {
		_current = (_current + 1) % 2;
	}
}



///////////////////////////


blib::Volume::Volume() 
{
	//Todo move somewhere else
	if (!glewInited) {
		auto glewCode = glewInit();
		if (glewCode != GLEW_OK)
			throw glewGetErrorString(glewCode);
		glewInited = true;
	}
	
}


void blib::Volume::addChannel(ivec3 dim, PrimitiveType type)
{
	_channels.push_back(VolumeChannel(dim, type));
}

VolumeChannel & blib::Volume::getChannel(uint index)
{
	assert(index < _channels.size());
	return _channels[index];
}

const VolumeChannel & blib::Volume::getChannel(uint index) const
{
	assert(index < _channels.size());
	return _channels[index];
}


void blib::Volume::erode(uint channel)
{
	auto & c = getChannel(channel);
	launchErodeKernel(
		make_uint3(c.dim.x, c.dim.y, c.dim.z),
		c.getCurrentPtr().getSurface(), //in
		c.getNextPtr().getSurface() //out
	);
	_CUDA(cudaDeviceSynchronize());

}

void blib::Volume::heat(uint channel)
{
	auto & c = getChannel(channel);
	launchHeatKernel(
		make_uint3(c.dim.x, c.dim.y, c.dim.z),
		c.getCurrentPtr().getSurface(), //in
		c.getNextPtr().getSurface() //out
	);
	_CUDA(cudaDeviceSynchronize());
}
