#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#pragma warning(disable:4554)  
#include <unsupported/Eigen/CXX11/Tensor>

#include "DataPtr.h"
#include "PrimitiveTypes.h"

#include <array>

namespace blib{

	


	struct VolumeChannel {

		BLIB_EXPORT VolumeChannel(ivec3 dim, PrimitiveType type, bool doubleBuffered = true);
		const ivec3 dim;
		const PrimitiveType type;

		BLIB_EXPORT Texture3DPtr & getCurrentPtr();
		BLIB_EXPORT Texture3DPtr & getNextPtr();
		BLIB_EXPORT const Texture3DPtr & getCurrentPtr() const;
		BLIB_EXPORT const Texture3DPtr & getNextPtr() const;

		BLIB_EXPORT void swapBuffers();
		
		
		

	private:		
		bool _doubleBuffered;
		uchar _current;
		std::array<Texture3DPtr, 2> _ptr;
		
	};

	struct Volume {
		BLIB_EXPORT Volume();

		BLIB_EXPORT void addChannel(ivec3 dim, PrimitiveType type);
		BLIB_EXPORT size_t emplaceChannel(VolumeChannel && channel) {
			_channels.emplace_back(std::move(channel));
			return _channels.size() - 1;
		}
		BLIB_EXPORT VolumeChannel & getChannel(uint index);
		BLIB_EXPORT const VolumeChannel & getChannel(uint index) const;

		
		BLIB_EXPORT void erode(uint channel);
		BLIB_EXPORT void heat(uint channel);
				
		BLIB_EXPORT void binarize(uint channel, float threshold = 1.0f);

		BLIB_EXPORT void diffuse(
			uint diffusivityChannel,
			uint concetrationChannel
			);

		

	private:
		std::vector<VolumeChannel> _channels;
		
	};	

}

/*

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

	template <typename T>
	Volume<T> diffuse(const Volume<T> & v, double diffusivity) {
		Volume<T> vnew = v;
					
		const auto dims = v.dimensions();

		//float h[3] = {1.0 / dims[0], 1.0 / dims[1], 1.0 / dims[2] };
		float h[3] = { 1,1,1 };


		#pragma omp parallel for schedule(dynamic)
		for (__int64 x = 1LL; x < dims[0] - 1LL; x++) {
			for (__int64 y = 1LL; y < dims[1] - 1LL; y++) {
				for (__int64 z = 1LL; z < dims[2] - 1LL; z++) {
					
					float val = v(x, y, z);

					float ddx = (float(v(x + 1LL, y, z)) - 2.0f * val - float(v(x - 1LL, y, z))) / h[0];
					float ddy = (float(v(x, y + 1LL, z)) - 2.0f * val - float(v(x, y - 1LL, z))) / h[1];
					float ddz = (float(v(x, y, z + 1LL)) - 2.0f * val - float(v(x, y, z - 1LL))) / h[2];


					float newVal = val + diffusivity * (ddx + ddy + ddz);
					if (newVal < 0.0f)
						vnew(x, y, z) = 0;
					else
						vnew(x, y, z) = T(newVal);
				}
			}
		}

		return vnew;

	}


}
*/

