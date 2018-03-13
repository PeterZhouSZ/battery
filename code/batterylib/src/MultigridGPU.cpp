#include "MultigridGPU.h"

#include "cuda/Volume.cuh"
#include "cuda/MultigridGPU.cuh"

using namespace blib;

template class MultigridGPU<float>;
template class MultigridGPU<double>;


template <typename T>
MultigridGPU<T>::MultigridGPU(bool verbose)
	: _verbose(verbose),
	_type((std::is_same<float, T>::value) ? TYPE_FLOAT : TYPE_DOUBLE)
{


}

template <typename T>
bool ::MultigridGPU<T>::prepare(
	Volume & volume,
	const VolumeChannel & mask, 
	const VolumeChannel & concetration, 
	Dir dir, 
	T d0, T d1, 
	uint levels, 
	vec3 cellDim
)
{

	
	_cellDim = cellDim;
	_iterations = 0;
	_lv = levels;

	

	

	//Prepare lin. system for all levels
	_A.resize(_lv);
	_f.resize(_lv);
	_x.resize(_lv);
	//_tmpx.resize(_lv);
	_r.resize(_lv);
	_dims.resize(_lv);

	//Generate D volume for smaller resolutions
	_D.resize(_lv);




	auto origDim = mask.dim();
	const size_t origTotal = origDim.x * origDim.y * origDim.z;

	{

		_dims.front() = origDim;
		auto & firstLevel = _D.front();
		firstLevel.allocOpenGL(_type, origDim, true);		


		launchConvertMaskKernel(
			_type,
			make_uint3(origDim.x,origDim.y,origDim.z),
			mask.getCurrentPtr().getSurface(), 
			firstLevel.getSurface(),
			d0, d1
		);

		//firstLevel.retrieve();

		cudaDeviceSynchronize();

		volume.emplaceChannel(
			VolumeChannel(firstLevel, origDim)
		);
		

		//res &= prepareAtLevelFVM(_D[0].data(), _dims[0], dir, 0);		
	}
	


	return true;
}


template <typename T>
T MultigridGPU<T>::solve(T tolerance, size_t maxIterations)
{
	return 0;
}