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


uint blib::Volume::addChannel(ivec3 dim, PrimitiveType type)
{
	_channels.push_back(VolumeChannel(dim, type));
	return static_cast<uint>(_channels.size() - 1);
}

uint blib::Volume::emplaceChannel(VolumeChannel && channel)
{
	_channels.emplace_back(std::move(channel));
	return static_cast<uint>(_channels.size() - 1);
}

VolumeChannel & blib::Volume::getChannel(uint index)
{
	assert(index < _channels.size());
	return _channels[index];
}

bool blib::Volume::hasChannel(uint index) const
{
	return index < _channels.size();
}

BLIB_EXPORT uint blib::Volume::numChannels() const
{
	return static_cast<uint>(_channels.size());
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

 void blib::Volume::binarize(uint channel, float threshold /*= 1.0f*/)
{
	 auto & c = getChannel(channel);
	 launchBinarizeKernel(
		 make_uint3(c.dim.x, c.dim.y, c.dim.z),
		 c.getCurrentPtr().getSurface(),
		 c.type,
		 threshold		 
	 );
	 _CUDA(cudaDeviceSynchronize());
}

 void blib::Volume::diffuse(uint maskChannel, uint concetrationChannel, float zeroDiff, float oneDiff)
 {
	 auto & cmask = getChannel(maskChannel);
	 auto & cconc = getChannel(concetrationChannel);

	 //Must have same dimensions
	 assert(cmask.dim.x == cconc.dim.x && cmask.dim.y == cconc.dim.y && cmask.dim.z == cconc.dim.z);
	 assert(zeroDiff >= 0.0f && oneDiff >= 0.0f);


	 launchDiffuseKernel(
		{
		 make_uint3(cmask.dim.x, cmask.dim.y, cmask.dim.z),
		 cmask.getCurrentPtr().getSurface(),
		 cconc.getCurrentPtr().getSurface(),
		 cconc.getNextPtr().getSurface(),
		 zeroDiff,
		 oneDiff,
		 //Boundary values
		 {			 
			 0.0f, 1.0f,
			 BOUNDARY_ZERO_GRADIENT, BOUNDARY_ZERO_GRADIENT,			 
			 BOUNDARY_ZERO_GRADIENT, BOUNDARY_ZERO_GRADIENT,			 
		 }
		}
	 );

	 _CUDA(cudaDeviceSynchronize());


 }

