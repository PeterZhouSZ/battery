#pragma once

#include "BatteryLibDef.h"

#include "DataPtr.h"

#include <array>
#include <vector>

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

		BLIB_EXPORT uint addChannel(ivec3 dim, PrimitiveType type);
		BLIB_EXPORT uint emplaceChannel(VolumeChannel && channel);
		BLIB_EXPORT VolumeChannel & getChannel(uint index);
		BLIB_EXPORT const VolumeChannel & getChannel(uint index) const;
		BLIB_EXPORT bool hasChannel(uint index) const;

		
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
