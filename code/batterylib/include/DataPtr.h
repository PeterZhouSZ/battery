#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#include <cuda_runtime.h>


namespace blib {

	struct DataPtr {
		void * cpu = nullptr;
		void * gpu = nullptr;		
		size_t stride = 0; //byte stride between elements		
		
		size_t num;
		BLIB_EXPORT size_t byteSize() const { return num*stride; }

		
		//Commit to device (offset in bytes)
		BLIB_EXPORT bool commit(size_t offset, size_t size);
		//Commits all
		BLIB_EXPORT bool commit();

		//Retrieve from device (offset in bytes)
		BLIB_EXPORT bool retrieve(size_t offset, size_t size);
		//Retrieves all
		BLIB_EXPORT bool retrieve();

		//Simple alloc
		BLIB_EXPORT bool allocHost(size_t num, size_t stride);
		BLIB_EXPORT bool allocDevice(size_t num, size_t stride);

		//Allocates both host and device memory
		BLIB_EXPORT bool alloc(size_t num, size_t stride);

		BLIB_EXPORT ~DataPtr();

	};

	/*
		TODO!!!! destructor, free gpu (and cpu) resources
	*/
	struct Texture3DPtr {	
		BLIB_EXPORT Texture3DPtr();
		BLIB_EXPORT ~Texture3DPtr();

		BLIB_EXPORT Texture3DPtr(const Texture3DPtr &) = delete;
		BLIB_EXPORT Texture3DPtr & operator = (const Texture3DPtr &) = delete;

		BLIB_EXPORT Texture3DPtr(Texture3DPtr &&other);
		BLIB_EXPORT Texture3DPtr & operator = (Texture3DPtr &&other);
				
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
			Copies data to linear global memory on device
		*/
		BLIB_EXPORT bool copyTo(DataPtr & ptr);

		BLIB_EXPORT bool copyFrom(DataPtr & ptr);

		/*
			Clears both cpu & gpu with val
			TODO: memset on gpu instead of doing cpu->gpu copy (i.e. using kernel/memset3d)
		*/
		BLIB_EXPORT bool clear(uchar val = 0);
		BLIB_EXPORT bool clearGPU(uchar val = 0);

		//Fills volume with elem of type primitivetype
		BLIB_EXPORT bool fillSlow(void * elem);

		BLIB_EXPORT PrimitiveType type() const { return _type; }
		
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