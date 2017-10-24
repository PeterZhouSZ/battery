#pragma once

#include <vector>
#include "utility/mathtypes.h"

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
};