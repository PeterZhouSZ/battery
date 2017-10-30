#pragma once

#include "BatteryLibDef.h"


#pragma warning(disable:4554)  
#include <unsupported/Eigen/CXX11/Tensor>

namespace blib {

	template <typename T>
	using Volume = Eigen::Tensor<T, 3>;

	BLIB_EXPORT Volume<unsigned char> emptyVolume(int size);

}
