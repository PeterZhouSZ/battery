#include "Volume.h"
#include "GLGlobal.h"

#include "Volume.cuh"
#include "CudaUtility.h"

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

using namespace blib;

bool glewInited = false;

blib::VolumeChannel::VolumeChannel(ivec3 dim, PrimitiveType type, bool doubleBuffered)
	: _dim(dim), _type(type), _current(0), _doubleBuffered(doubleBuffered)
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

 void blib::VolumeChannel::resize(ivec3 origin, ivec3 newDim)
{
	 
	 const ivec3 end = origin + newDim;
	 assert(end.x <= _dim.x);
	 assert(end.y <= _dim.y);
	 assert(end.z <= _dim.z);
 

	 VolumeChannel tmp = VolumeChannel(newDim, _type, _doubleBuffered);

	 for (auto i = 0; i < ((_doubleBuffered) ? 2 : 1); i++) {
		
		 
		 const auto & p0 = _ptr[i];
		 const auto stride = p0.stride();
		 const uchar * data0 = reinterpret_cast<const uchar *>(p0.getCPU());

		 auto & p1 = tmp._ptr[i];
		 uchar * data1 = reinterpret_cast<uchar *>(p1.getCPU());

		 for (auto z = origin.z; z < end.z; z++) {
			 for (auto y = origin.y; y < end.y; y++) {				 
				 auto x = origin.x;
				 auto i0 = linearIndex(_dim, x, y, z);
				 auto i1 = linearIndex(newDim, x - origin.x, y - origin.y, z - origin.z);
				 memcpy(data1 + stride * i1, data0 + stride * i0, stride * newDim.x);				 
			 }
		 }		 

		 p1.commit();

	 }
	 *this = std::move(tmp);

}

float blib::VolumeChannel::differenceSum()
{

	//Only float supported at the moment
	assert(type() == TYPE_FLOAT);

	auto res = make_uint3(dim().x, dim().y, dim().z);
	launchSubtractKernel(
		res,
		getCurrentPtr().getSurface(),
		getNextPtr().getSurface()
	);	
		
	float result = launchReduceSumKernel(res, getCurrentPtr().getSurface());				

	return result;
}

void blib::VolumeChannel::clear()
{
	clearCurrent();
	if(_doubleBuffered)
		clearNext();
}

void blib::VolumeChannel::clearCurrent()
{
	getCurrentPtr().clear(0);	
}

void blib::VolumeChannel::clearNext()
{
	getNextPtr().clear(0);
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



uint blib::VolumeChannel::dimInDirection(Dir dir)
{
	return dim()[getDirIndex(dir)];
}

uint blib::VolumeChannel::sliceElemCount(Dir dir)
{
	uint index = getDirIndex(dir);
	return dim()[(index + 1) % 3] * dim()[(index + 2) % 3];
}

bool blib::VolumeChannel::isDoubleBuffered() const
{
	return _doubleBuffered;
}

ivec3 blib::VolumeChannel::dim() const
{
	return _dim;
}

PrimitiveType blib::VolumeChannel::type() const
{
	return _type;
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

uint blib::Volume::emplaceChannel(VolumeChannel && channel, uint index)
{
	if (index >= numChannels()) {
		_channels.push_back(std::move(channel));
		return static_cast<uint>(_channels.size() - 1);
	}
	else {		
		_channels[index] = std::move(channel);		
		return index;
	}
	
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
		make_uint3(c.dim().x, c.dim().y, c.dim().z),
		c.getCurrentPtr().getSurface(), //in
		c.getNextPtr().getSurface() //out
	);


}

void blib::Volume::heat(uint channel)
{
	auto & c = getChannel(channel);
	launchHeatKernel(
		make_uint3(c.dim().x, c.dim().y, c.dim().z),
		c.getCurrentPtr().getSurface(), //in
		c.getNextPtr().getSurface() //out
	);
	
}

 void blib::Volume::binarize(uint channel, float threshold /*= 1.0f*/)
{
	 auto & c = getChannel(channel);
	 launchBinarizeKernel(
		 make_uint3(c.dim().x, c.dim().y, c.dim().z),
		 c.getCurrentPtr().getSurface(),
		 c.type(),
		 threshold		 
	 );
	 
}

 void blib::Volume::reduceSlice(uint channel, Dir dir, void * output)
 {
	 auto & c = getChannel(channel);
	 
	 //Only float supported at the moment
	 assert(c.type() == TYPE_FLOAT);

  	 launchReduceSumSlice(
		 make_uint3(c.dim().x, c.dim().y, c.dim().z),
		 c.getCurrentPtr().getSurface(),
		 dir,
		 output
	 );

 }

 void blib::Volume::diffuse(
	 uint maskChannel,
	 uint concetrationChannel, 
	 float voxelSize,
	 float zeroDiff, 
	 float oneDiff,
	 float highConc,
	 float lowConc,
	 Dir diffusionDir
 )
 {
	 auto & cmask = getChannel(maskChannel);
	 auto & cconc = getChannel(concetrationChannel);

	 //Must have same dimensions
	 assert(cmask.dim().x == cconc.dim().x && cmask.dim().y == cconc.dim().y && cmask.dim().z == cconc.dim().z);
	 assert(zeroDiff >= 0.0f && oneDiff >= 0.0f);

	 std::array<float, 6> boundary;
	 for (auto &v : boundary)
		 v = BOUNDARY_ZERO_GRADIENT;
	 
	 switch (diffusionDir) {
	 case X_POS:
		 boundary[X_NEG] = lowConc;
		 boundary[X_POS] = highConc;
		 break;
	 case X_NEG:
		 boundary[X_POS] = lowConc;
		 boundary[X_NEG] = highConc;
		 break;
	 case Y_POS:
		 boundary[Y_NEG] = lowConc;
		 boundary[Y_POS] = highConc;
		 break;
	 case Y_NEG:
		 boundary[Y_POS] = lowConc;
		 boundary[Y_NEG] = highConc;
		 break;
	 case Z_POS:
		 boundary[Z_NEG] = lowConc;
		 boundary[Z_POS] = highConc;
		 break;
	 case Z_NEG:
		 boundary[Z_POS] = lowConc;
		 boundary[Z_NEG] = highConc;
		 break;
	 }
	 
	 
	 

	 launchDiffuseKernel(
		{
		 make_uint3(cmask.dim().x, cmask.dim().y, cmask.dim().z),
		 voxelSize,
		 cmask.getCurrentPtr().getSurface(),
		 cconc.getCurrentPtr().getSurface(),
		 cconc.getNextPtr().getSurface(),
		 zeroDiff,
		 oneDiff,
		 //Boundary values
		 {
			 boundary[0],boundary[1],boundary[2],
			 boundary[3],boundary[4],boundary[5]
		 }
		}
	 );

	 cudaDeviceSynchronize();


 }


