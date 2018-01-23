#pragma once

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
		void * getCPU() { return _cpu.ptr; }
		const void * getCPU() const { return _cpu.ptr; }

		/*
			OpenGL ID of texture
		*/
		uint getGlID() const { return _glID; }
			
		/*
			Total number of elements
		*/
		uint64 num() const { 
			return _extent.width * _extent.height * _extent.depth;
		}

		/*
			Total bytesize
		*/
		uint64 byteSize() const { 
			return num() * stride(); 
		}

		/*
			Size of element
		*/
		uint64 stride() const {
			return _desc.x + _desc.y + _desc.z + _desc.w;
		}

		ivec3 dim() const {
			return { _extent.width, _extent.height, _extent.depth };
		}

		/*
			Allocates 3D array 
		*/
		bool alloc(PrimitiveType type, ivec3 dim, bool alsoOnCPU = false);

		/*
			Allocates 3D array using OpenGL interop
		*/
		bool allocOpenGL(PrimitiveType type, ivec3 dim, bool alsoOnCPU = false);

				
		/*
			Commits host memory to device
		*/
		bool commit();

		/*
			Retrieves device memory to host
		*/
		bool retrieve();

		
		/*
			Returns cuda surface handle
		*/
		cudaSurfaceObject_t getSurface() const { 
			return _surface; 
		}

		/*
			Copies cuda surface handle to specified device memory
		*/
		bool copySurfaceTo(void * gpuSurfacePtr) const;
		
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
		
	};

}