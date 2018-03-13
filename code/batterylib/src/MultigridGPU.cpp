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

	
	_D.resize(_lv);

	//Calculate dimensions for all levels
	_dims[0] = mask.dim();	
	for (uint i = 1; i < _lv; i++) {
		ivec3 dim = _dims[i - 1] / 2;
		if (_dims[i - 1].x % 2 == 1) dim.x++;
		if (_dims[i - 1].y % 2 == 1) dim.y++;
		if (_dims[i - 1].z % 2 == 1) dim.z++;
		_dims[i] = dim;
	}	

	//Convert mask to float/double 
	{		
		auto & firstLevel = _D[0];
		firstLevel.allocOpenGL(_type, _dims[0], false);
		
		launchConvertMaskKernel(
			_type,
			make_uint3(_dims[0].x, _dims[0].y, _dims[0].z),
			mask.getCurrentPtr().getSurface(), 
			firstLevel.getSurface(),
			d0, d1
		);					

		cudaDeviceSynchronize();
		volume.emplaceChannel(
			VolumeChannel(firstLevel, _dims[0], "D in float/double")
		);		
	}

	//Restrict diffusion coeffs for smaller grids
	for (uint i = 1; i < _lv; i++) {
				
		_D[i].allocOpenGL(_type, _dims[i], false);
				
		launchRestrictionKernel(
			_type,
			_D[i - 1].getSurface(),
			make_uint3(_dims[i - 1].x, _dims[i - 1].y, _dims[i - 1].z),
			_D[i].getSurface(),
			make_uint3(_dims[i].x, _dims[i].y, _dims[i].z),
			T(0.5)
		);

		
		volume.emplaceChannel(
			VolumeChannel(_D[i], _dims[i])
		);


	}
	cudaDeviceSynchronize();

	


	return true;
}


template <typename T>
T MultigridGPU<T>::solve(T tolerance, size_t maxIterations)
{
	return 0;
}