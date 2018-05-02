
# Build instructions:

## Libraries required:
- [GLFW](http://www.glfw.org/) *>= 3.2.1*  
- [GLEW](http://glew.sourceforge.net/) *>= 2.0.0*
- [GLM](https://glm.g-truc.net/0.9.8/index.html) *>= 0.9.8.x*
- [EIGEN](http://eigen.tuxfamily.org/index.php?title=Main_Page) *>= 3.3.4*
- [CUDA](https://developer.nvidia.com/cuda-downloads) (tested on 9.0)


### Environment variables

- **GLFW_PATH** to GLFW root directory
- **GLEW_PATH** to GLEW root directory, **static** build
- **GLM_PATH** to GLM root directory
- **EIGEN_PATH** to EIGEN root directory

## Build tools:
- [Premake5](https://premake.github.io/download.html)
- [premake-cuda](https://github.com/krsvojte/premake-cuda) script (included)


## Build steps (Windows):
- Download & compile libraries
- Set environment variables 
- In case of different version of CUDA than 9.0, change "cuda_version" variable in premake5.lua to your version.
- Run commmand 
```/build/premake5(.exe) vs2017```  
  - or use provided .bat files


# Projects
- **batterylib** | core data types and functionality, including tortuosity calculator
- **batteryviz** | graphical user interface and a volume visualization engine
- **batterytool** | command line interface for battery data analysis

