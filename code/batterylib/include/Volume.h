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


	template <typename T>
	Volume<T> erode(const Volume<T> & v) {

		Volume<T> vnew;
		const auto s = v.dimensions();
		vnew.resize(s[0], s[1], s[2]);
		vnew.setConstant(std::numeric_limits<T>::max());

		const auto dims = v.dimensions();
		#pragma omp parallel for schedule(dynamic)
		for (__int64 x = 0; x < dims[0]; x++) {
			for (__int64  y = 0; y < dims[1]; y++) {
				for (__int64  z = 0; z < dims[2]; z++) {

					for (__int64  i = std::max(x - 1LL, 0LL); i <= std::min(x + 1LL, dims[0] - 1LL); i++) {
						for (__int64  j = std::max(y - 1LL, 0LL); j <= std::min(y + 1LL, dims[1] - 1LL); j++) {
							for (__int64  k = std::max(z - 1LL, 0LL); k <= std::min(z + 1LL, dims[2] - 1LL); k++) {
								vnew(x, y, z) = std::min(v(i, j, k), vnew(x, y, z));
							}
						}
					}
				}
			}
		}

		
		return vnew;
	}

	template <typename T>
	Volume<T> dilate(const Volume<T> & v) {

		Volume<T> vnew;
		const auto s = v.dimensions();
		vnew.resize(s[0], s[1], s[2]);
		vnew.setConstant(std::numeric_limits<T>::min());

		const auto dims = v.dimensions();
		#pragma omp parallel for schedule(dynamic)
		for (__int64 x = 0; x < dims[0]; x++) {
			for (__int64 y = 0; y < dims[1]; y++) {
				for (__int64 z = 0; z < dims[2]; z++) {

					for (__int64 i = std::max(x - 1LL, 0LL); i <= std::min(x + 1LL, dims[0] - 1LL); i++) {
						for (__int64 j = std::max(y - 1LL, 0LL); j <= std::min(y + 1LL, dims[1] - 1LL); j++) {
							for (__int64 k = std::max(z - 1LL, 0LL); k <= std::min(z + 1LL, dims[2] - 1LL); k++) {
								vnew(x, y, z) = std::max(v(i, j, k), vnew(x, y, z));
							}
						}
					}
				}
			}
		}


		return vnew;
	}

}
