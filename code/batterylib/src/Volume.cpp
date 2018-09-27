#include "Volume.h"
#include "GLGlobal.h"

#include "Volume.cuh"
#include "CudaUtility.h"

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include "../src/cuda/VolumeTypes.cuh"

using namespace blib;

bool glewInited = false;

blib::VolumeChannel::VolumeChannel(ivec3 dim, PrimitiveType type, bool doubleBuffered, const std::string & name)
	: _dim(dim), _type(type), _current(0), _doubleBuffered(doubleBuffered),
	_name(name)
{	


	//Allocate buffer(s)
	if (VolumeChannel::enableOpenGLInterop) {
		_ptr[_current].allocOpenGL(type, dim, true);
		if (doubleBuffered) {
			_ptr[(_current + 1) % 2].allocOpenGL(type, dim, true);
		}
	}
	else {
		_ptr[_current].alloc(type, dim, true);
		if (doubleBuffered) {
			_ptr[(_current + 1) % 2].alloc(type, dim, true);
		}
	}
	

	
	//Test fill
	auto & ptr = getCurrentPtr();	
	float * arr = reinterpret_cast<float*>(ptr.getCPU());	
	memset(arr, 0, ptr.byteSize());
	ptr.commit();
}


blib::VolumeChannel::VolumeChannel(Texture3DPtr && ptr, ivec3 dim, const std::string & name)
	: _ptr{std::move(ptr)}, _dim(dim), _type(ptr.type()), _doubleBuffered(false), _name(name), _current(0)
{

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

void blib::VolumeChannel::sum(void * result)
{
	assert(result != nullptr);

	const size_t reduceN = Volume_Reduce_RequiredBufferSize(dim().x * dim().y * dim().z);

	PrimitiveType reductionType = _type;
	if (_type == TYPE_CHAR || _type == TYPE_UCHAR || _type == TYPE_INT || _type == TYPE_UINT) {
		reductionType = TYPE_UINT64;
	}
	
	DataPtr aux;
	aux.alloc(reduceN, primitiveSizeof(reductionType));	

	Volume_Reduce(*getCUDAVolume(), REDUCE_OP_SUM, reductionType, aux.gpu, aux.cpu, result);

}

size_t blib::VolumeChannel::nonZeroElems() const
{
	const size_t reduceN = Volume_Reduce_RequiredBufferSize(dim().x * dim().y * dim().z);
	DataPtr aux;
	aux.alloc(reduceN, primitiveSizeof(TYPE_UINT64));	

	uint64 result;
	Volume_Reduce(*getCUDAVolume(), REDUCE_OP_SUM_NONZERO, TYPE_UINT64, aux.gpu, aux.cpu, &result);

	return size_t(result);

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

template <typename T>
void _sumInDir(ivec3 dim, Dir dir, T * in, T * out){
	
	int primary = getDirIndex(dir);
	int secondary[2] = { (primary + 1) % 3, (primary + 2) % 3 };
	int sgn = getDirSgn(dir);

//	#pragma omp parallel
	for (auto i = 0; i < dim[primary]; i++) {

		T sum = 0.0f;
		for (auto j = 0; j < dim[secondary[0]]; j++) {
			for (auto k = 0; k < dim[secondary[1]]; k++) {

				ivec3 pos;
				pos[primary] = i;
				pos[secondary[0]] = j;
				pos[secondary[1]] = k;

				sum += in[linearIndex(dim, pos)];
			}
		}

		if(sgn > 0)
			out[i] = sum;	
		else
			out[dim[primary] - 1 - i] = sum;
	}

}

void blib::VolumeChannel::sumInDir(Dir dir, void * output)
{
	auto & ptr = getCurrentPtr();
	ptr.allocCPU();
	ptr.retrieve();
		
	
	if (_type == TYPE_FLOAT) {
		_sumInDir<float>(_dim, dir, static_cast<float*>(ptr.getCPU()), static_cast<float*>(output));
	}
	else if (_type == TYPE_DOUBLE) {
		_sumInDir<double>(_dim, dir, static_cast<double*>(ptr.getCPU()), static_cast<double*>(output));
	}	
	else {
		assert("NOT IMPLEMENTED");
	}

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

size_t blib::VolumeChannel::totalElems() const
{
	return dim().x * dim().y * dim().z;
}

void blib::VolumeChannel::setName(const std::string & name)
{
	_name = name;
}

std::string blib::VolumeChannel::getName() const
{
	return _name;
}

void blib::VolumeChannel::normalize()
{

	size_t reduceN = (dim().x * dim().y * dim().z) / VOLUME_REDUCTION_BLOCKSIZE;
	if (reduceN < 512)
		reduceN = 512;

	DataPtr reduceBuffer;
	reduceBuffer.alloc(reduceN, primitiveSizeof(_type));


	char result[64];
	launchReduceKernel(
		_type,
		REDUCE_OP_MIN,
		make_uint3(dim().x, dim().y, dim().z),
		getCurrentPtr().getSurface(),
		reduceBuffer.gpu,
		reduceBuffer.cpu,
		result
	);

	double minimum = primitiveToDouble(_type, result);

	

	launchReduceKernel(
		_type,
		REDUCE_OP_MAX,
		make_uint3(dim().x, dim().y, dim().z),
		getCurrentPtr().getSurface(),
		reduceBuffer.gpu,
		reduceBuffer.cpu,
		result
	);

	double maximum = primitiveToDouble(_type, result);

	launchNormalizeKernel(_type, getCurrentPtr().getSurface(), make_uint3(dim().x, dim().y, dim().z), minimum, maximum);

	{
		char result[64];
		launchReduceKernel(
			_type,
			REDUCE_OP_MIN,
			make_uint3(dim().x, dim().y, dim().z),
			getCurrentPtr().getSurface(),
			reduceBuffer.gpu,
			reduceBuffer.cpu,
			result
		);

		double minimum = primitiveToDouble(_type, result);



		launchReduceKernel(
			_type,
			REDUCE_OP_MAX,
			make_uint3(dim().x, dim().y, dim().z),
			getCurrentPtr().getSurface(),
			reduceBuffer.gpu,
			reduceBuffer.cpu,
			result
		);

		double maximum = primitiveToDouble(_type, result);

		std::cout << "[" << minimum << ", " << maximum << "]" << std::endl;
	
	}

	


}

std::shared_ptr<CUDA_Volume> blib::VolumeChannel::getCUDAVolume() const
{
	auto v = std::make_shared<CUDA_Volume>();	
	v->ID = -1;
	v->surf = getCurrentPtr().getSurface();
	v->res = make_uint3(dim().x, dim().y, dim().z);
	v->type = type();	
	v->tex = getCurrentPtr().getTexture();
	return v;
}

bool blib::VolumeChannel::enableOpenGLInterop = false;

///////////////////////////


blib::Volume::Volume() 
{
	//Todo move somewhere else
	if (VolumeChannel::enableOpenGLInterop && !glewInited) {
		auto glewCode = glewInit();
		if (glewCode != GLEW_OK)
			throw glewGetErrorString(glewCode);
		glewInited = true;
	}
	
}

std::shared_ptr<CUDA_Volume> blib::Volume::getCUDAVolume(int ID) const
{
	auto v = std::make_shared<CUDA_Volume>();
	auto &c = getChannel(ID);
	v->ID = ID;
	v->surf = c.getCurrentPtr().getSurface();
	v->res = make_uint3(c.dim().x, c.dim().y, c.dim().z);
	v->type = c.type();
	v->cpu = nullptr;
	v->tex = c.getCurrentPtr().getTexture();
	return v;
}

std::shared_ptr<CUDA_Volume> blib::Volume::getCUDAVolume(int ID)
{
	auto v = std::make_shared<CUDA_Volume>();
	auto &c = getChannel(ID);
	v->ID = ID;
	v->surf = c.getCurrentPtr().getSurface();
	v->res = make_uint3(c.dim().x, c.dim().y, c.dim().z);
	v->type = c.type();
	v->cpu = c.getCurrentPtr().getCPU();
	v->tex = c.getCurrentPtr().getTexture();
	return v;
}

uint blib::Volume::addChannel(ivec3 dim, PrimitiveType type, bool doubleBuffered, const std::string & name)
{
	_channels.push_back(
		std::make_shared<VolumeChannel>(dim, type, doubleBuffered, name)
	);
	return static_cast<uint>(_channels.size() - 1);
}

uint blib::Volume::emplaceChannel(VolumeChannel && channel, uint index)
{
	if (index >= numChannels()) {
		_channels.push_back(std::make_shared<VolumeChannel>(std::move(channel)));
		//_channels.push_back(std::move(channel));
		return static_cast<uint>(_channels.size() - 1);
	}
	else {		
		*_channels[index] = std::move(channel);		
		return index;
	}
	
}

VolumeChannel & blib::Volume::getChannel(uint index)
{
	assert(index < _channels.size());
	return *_channels[index];
}

bool blib::Volume::hasChannel(uint index) const
{
	return index < _channels.size();
}

BLIB_EXPORT uint blib::Volume::numChannels() const
{
	return static_cast<uint>(_channels.size());
}

bool blib::Volume::removeChannel(uint index)
{
	assert(index < _channels.size());
	_channels.erase(_channels.begin() + index);
	return true;
}

const VolumeChannel & blib::Volume::getChannel(uint index) const
{
	assert(index < _channels.size());
	return *_channels[index];
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

 BLIB_EXPORT void blib::Volume::synchronize()
 {
	 cudaDeviceSynchronize();
 }

 BLIB_EXPORT void blib::Volume::clear()
 {
	 _channels.clear();
 }

