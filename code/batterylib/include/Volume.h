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

		
		//Call before swap buffers
		//Current = Current - Next; Sum(Current);
		BLIB_EXPORT float differenceSum();

		/*
			Clears both buffers
		*/
		BLIB_EXPORT void clear();
		BLIB_EXPORT void clearCurrent();
		BLIB_EXPORT void clearNext();

		BLIB_EXPORT void swapBuffers();
		
		BLIB_EXPORT uint dimInDirection(Dir dir);
		BLIB_EXPORT uint sliceElemCount(Dir dir);
		

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
		BLIB_EXPORT uint numChannels() const;
		
		BLIB_EXPORT void erode(uint channel);
		BLIB_EXPORT void heat(uint channel);
				
		BLIB_EXPORT void binarize(uint channel, float threshold = 1.0f);
		
		//Reduces for each slice in dir and writes result to output
		BLIB_EXPORT void reduceSlice(uint channel, Dir dir, void * output);

		BLIB_EXPORT void diffuse(
			uint maskChannel,
			uint concetrationChannel,
			float voxelSize,
			float zeroDiff,
			float oneDiff,
			float highConc = 1.0f,
			float lowConc = 0.0f,
			Dir diffusionDir = X_POS
			);

		

	private:
		std::vector<VolumeChannel> _channels;
		
	};	

}
