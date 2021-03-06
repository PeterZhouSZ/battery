#pragma once

#include "BatteryLibDef.h"
#include "Types.h"

#include <cuda_runtime.h>



namespace blib {

	
	

	struct DataPtr {
		void * cpu;
		void * gpu;
		size_t stride; //byte stride between elements		
		
		size_t num;

		BLIB_EXPORT DataPtr();

		BLIB_EXPORT DataPtr(const DataPtr &) = delete;
		BLIB_EXPORT DataPtr & operator = (const DataPtr &) = delete;

		BLIB_EXPORT DataPtr(DataPtr &&other);
		BLIB_EXPORT DataPtr & operator = (DataPtr &&other);

		BLIB_EXPORT size_t byteSize() const { return num*stride; }

		BLIB_EXPORT bool memsetDevice(int value = 0);
		
		//Commit to device (offset in bytes)
		BLIB_EXPORT bool commit(size_t offset, size_t size);
		//Commits all
		BLIB_EXPORT bool commit();

		//Retrieve from device (offset in bytes)
		BLIB_EXPORT bool retrieve(size_t offset, size_t size);
		//Retrieves all
		BLIB_EXPORT bool retrieve();

		//Simple alloc
		BLIB_EXPORT bool allocHost();
		BLIB_EXPORT bool allocDevice(size_t num, size_t stride);

		//Allocates both host and device memory
		BLIB_EXPORT bool alloc(size_t num, size_t stride);

		BLIB_EXPORT ~DataPtr();

	private:
		void _free();

	};

	
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
			return (_desc.x  + _desc.y + _desc.z + _desc.w) / 8;
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

		BLIB_EXPORT bool allocCPU();

				
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

		BLIB_EXPORT cudaTextureObject_t getTexture() const {
			return _texture;
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

		BLIB_EXPORT bool createTexture();


		BLIB_EXPORT bool hasTextureObject() const {
			return _textureCreated;
		}
		
	private:

		void _free();
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

		bool _textureCreated;
		cudaTextureObject_t _texture;

		cudaChannelFormatDesc _desc;
		cudaExtent _extent;
		uint _glID;
		PrimitiveType _type;	

		bool _usesOpenGL;
		
	};

	struct CUDA_VBO {

		BLIB_EXPORT CUDA_VBO(uint vbo);
		BLIB_EXPORT ~CUDA_VBO();

		BLIB_EXPORT CUDA_VBO(const CUDA_VBO &) = delete;
		BLIB_EXPORT CUDA_VBO & operator = (const CUDA_VBO &) = delete;

		BLIB_EXPORT CUDA_VBO(CUDA_VBO &&other);
		BLIB_EXPORT CUDA_VBO & operator = (CUDA_VBO &&other);

		BLIB_EXPORT void * getPtr() {
			return _ptr;
		}

		BLIB_EXPORT const void * getPtr() const {
			return _ptr;
		}

		BLIB_EXPORT uint getVBO() const {
			return _vbo;
		}

		BLIB_EXPORT void retrieveTo(void * ptr) const;

		struct DefaultAttrib {
			float pos[3]; 
			float normal[3];
			float uv[2];
			float color[4];
		};


	private:

		void _free();
		uint _vbo;
		cudaGraphicsResource_t _resource;
		void * _ptr;
		size_t _bytes;		
	};

	/*
		Allocates an OpenGL VBO and maps it for CUDA.
		! The structure does not own the vbo, it must be destroyed manually. !
	*/
	BLIB_EXPORT CUDA_VBO createMappedVBO(size_t bytesize);

	

}