#include "VolumeSurface.h"

#include "Volume.h"
#include "cuda/VolumeSurface.cuh"

namespace blib {


	BLIB_EXPORT VolumeChannel blib::getVolumeArea(const VolumeChannel & mask, ivec3 res /*= ivec3(0,0,0)*/, float isovalue /*= 0.5f */)
	{

		if (!mask.getCurrentPtr().hasTextureObject()) {
			throw "Mask has to have a texture object. Create it by mask.getCurrentPtr().createTextureObject()";
		}

		VolumeSurface_MCParams params;
		if (res.x == 0 || res.y == 0 || res.z == 0) {
			res = mask.dim();
		}
		
		
		params.res = make_uint3(res.x, res.y, res.z);
		params.isovalue = isovalue;
		params.smoothingOffset = 1.0f;

		VolumeChannel areas(res, TYPE_DOUBLE, false, "Area sum");
		VolumeSurface_MarchingCubesArea(*mask.getCUDAVolume(), params, *areas.getCUDAVolume());


		return areas;
		
	}

	BLIB_EXPORT bool getVolumeAreaMesh(const VolumeChannel & mask, uint * vboOut, size_t * NvertsOut, ivec3 res /*= ivec3(0, 0, 0)*/, float isovalue /*= 0.5f*/)
	{
		if (!mask.getCurrentPtr().hasTextureObject()) {
			throw "Mask has to have a texture object. Create it by mask.getCurrentPtr().createTextureObject()";
		}

		if (!NvertsOut || !vboOut) {
			throw "Nullptr output arguments";
		}

		VolumeSurface_MCParams params;
		if (res.x == 0 || res.y == 0 || res.z == 0) {
			res = mask.dim();
		}

		params.res = make_uint3(res.x, res.y, res.z);
		params.isovalue = isovalue;
		params.smoothingOffset = 1.0f;

		VolumeSurface_MarchingCubesMesh(*mask.getCUDAVolume(), params, vboOut, NvertsOut);

		
		return true;
	}

}

