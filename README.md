
# Build instructions:

## Libraries required:
- [GLFW](http://www.glfw.org/) *>= 3.2.1*  
- [GLEW](http://glew.sourceforge.net/) *>= 2.0.0*
- [GLM](https://glm.g-truc.net/0.9.8/index.html) *>= 0.9.8.x*
- [EIGEN](http://eigen.tuxfamily.org/index.php?title=Main_Page) *>= 3.3.4*
- [CUDA](https://developer.nvidia.com/cuda-downloads) (tested on 9.0+, 9.2 recommended)

### Compiler requirements
- C++17 compatible (Tested on VS 2017, Clang 6, GCC 7)


### Environment variables (Windows only)

- **GLFW_PATH** to GLFW root directory
- **GLEW_PATH** to GLEW root directory, **static** build
- **GLM_PATH** to GLM root directory
- **EIGEN_PATH** to EIGEN root directory

## Build tools:
- CMake 3.10+


## Build steps:
- Download & compile libraries
- Set environment variables (Windows only)
- Run cmake or cmake-gui

# Projects
- **batterylib** | core data types and functionality, including tortuosity calculator
- **batteryviz** | graphical user interface and a volume visualization engine
- **batterytool** | command line interface for battery data analysis

