#pragma once

#include "BatteryLibDef.h"


#pragma warning(disable:4554)  
#include <unsupported/Eigen/CXX11/Tensor>

namespace blib {

	template <typename T>
	using Volume = Eigen::Tensor<T, 3>;

	template <typename T>
	Volume<T> emptyVolume(int size) {
		Volume<T> v;
		v.resize(size, size, size);
		return v;
	}

}
