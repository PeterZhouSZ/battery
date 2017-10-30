#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#if defined(BATTERYLIB_EXPORT) // inside DLL
#   define BLIB_EXPORT   __declspec(dllexport)
#else // outside DLL
#   define BLIB_EXPORT   __declspec(dllimport)
#endif  // XYZLIBRARY_EXPORT


template <typename T>
using Volume = Eigen::Tensor<T, 3>;


BLIB_EXPORT Volume<unsigned char> emptyVolume(int size);

//TODO:
//Mesh to volume (scanline)
//Volume to mesh (marching cubes)
//Implicit function to volume

/*
template<typename T>
struct Volume {

	T & at(const ivec3 & coord) {
		//todo optim
		return data[coord.x + coord.y * size.x + coord.z * size.x * size.y];
	}

	T & at(int x, int y, int z) {
		return at({ x,y,z });
	}

	void clear(const T & val) {
		data.clear();
		data.resize(size.x * size.y * size.z);
	}

	void resize(const ivec3 & newSize, const T & val) {
		size = newSize;
		data.resize(newSize.x * newSize.y * newSize.z, val);
	}
	
	
	ivec3 size;
	std::vector<T> data;
};*/