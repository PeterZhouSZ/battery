#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#include <cuda_runtime.h>


namespace blib {

	struct DataPtr {
		void * cpu = nullptr;
		void * gpu = nullptr;		
		uint stride = 0; //byte stride between elements		
		
		uint num;
		uint byteSize() const { return num*stride; }

		
		//Commit to device (offset in bytes)
		bool commit(uint offset, uint size);
		//Commits all
		bool commit();		

		//Retrieve from device (offset in bytes)
		bool retrieve(uint offset, uint size);
		//Retrieves all
		bool retrieve();

		//Simple alloc
		bool allocHost(uint num, uint stride);				
		bool allocDevice(uint num, uint stride);

		~DataPtr();

	};


	struct Texture3DPtr {	
		Texture3DPtr();
				
		/*
			Returns host memory
		*/
		BLIB_EXPORT void * getCPU() { return _cpu.ptr; }
		BLIB_EXPORT const void * getCPU() const { return _cpu.ptr; }

		/*
			Returns GPU cudaArray
		*/
		BLIB_EXPORT const cudaArray * getGPUArray() const { return _gpu; }
		BLIB_EXPORT cudaArray * getGPUArray() { return _gpu; }
		BLIB_EXPORT bool mapGPUArray();
		BLIB_EXPORT bool unmapGPUArray();

		/*
			OpenGL ID of texture
		*/
		BLIB_EXPORT uint getGlID() const { return _glID; }
			
		/*
			Total number of elements
		*/
		BLIB_EXPORT uint64 num() const {
			return _extent.width * _extent.height * _extent.depth;
		}

		/*
			Total bytesize
		*/
		BLIB_EXPORT uint64 byteSize() const {
			return num() * stride(); 
		}

		/*
			Size of element
		*/
		BLIB_EXPORT uint64 stride() const {
			return _desc.x + _desc.y + _desc.z + _desc.w;
		}

		BLIB_EXPORT ivec3 dim() const {
			return { _extent.width, _extent.height, _extent.depth };
		}

		/*
			Allocates 3D array 
		*/
		BLIB_EXPORT bool alloc(PrimitiveType type, ivec3 dim, bool alsoOnCPU = false);

		/*
			Allocates 3D array using OpenGL interop
		*/
		BLIB_EXPORT bool allocOpenGL(PrimitiveType type, ivec3 dim, bool alsoOnCPU = false);

				
		/*
			Commits host memory to device
		*/
		BLIB_EXPORT bool commit();

		/*
			Retrieves device memory to host
		*/
		BLIB_EXPORT bool retrieve();

		
		/*
			Returns cuda surface handle
		*/
		BLIB_EXPORT cudaSurfaceObject_t getSurface() const {
			return _surface; 
		}

		/*
			Copies cuda surface handle to specified device memory
		*/
		BLIB_EXPORT bool copySurfaceTo(void * gpuSurfacePtr) const;


		/*
			Clears both cpu & gpu with val
			TODO: memset on gpu instead of doing cpu->gpu copy (i.e. using kernel/memset3d)
		*/
		BLIB_EXPORT bool clear(uchar val = 0);

		//Fills volume with elem of type primitivetype
		BLIB_EXPORT bool fillSlow(void * elem);
		
	private:

		/*
			Creates surface object
		*/
		bool createSurface();

		/*
			Sets channel description depending on type
		*/
		void setDesc(PrimitiveType type);

		cudaPitchedPtr _cpu;
		cudaArray * _gpu;
		cudaGraphicsResource * _gpuRes; //for GL  interop
		cudaSurfaceObject_t _surface;

		cudaChannelFormatDesc _desc;
		cudaExtent _extent;
		uint _glID;
		PrimitiveType _type;
		
	};

}