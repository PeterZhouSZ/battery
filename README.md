
# Build instructions:

## Libraries needed:
- [GLFW](http://www.glfw.org/) *>= 3.2.1*  
- [GLEW](http://glew.sourceforge.net/) *>= 2.0.0*
- [GLM](https://glm.g-truc.net/0.9.8/index.html) *>= 0.9.8.x*
- [EIGEN](http://eigen.tuxfamily.org/index.php?title=Main_Page) *>= 3.3.4*
- [CUDA](https://developer.nvidia.com/cuda-downloads) (tested on 9.0)

~~(Note: GLM will be replaced by Eigen in near future, Vojtech)~~

(Note: Eigen is replaced by GLM. Will keep Eigen dependency in case some CPU lin. algebra is needed)


## Build tools:
- [Premake5](https://premake.github.io/download.html)
- [premake-cuda](https://github.com/krsvojte/premake-cuda) script (included)

### Environment variables

- **GLFW_PATH** to GLFW root directory
- **GLEW_PATH** to GLEW root directory, **static** build
- **GLM_PATH** to GLM root directory
- **EIGEN_PATH** to EIGEN root directory


## Steps:
- Download & compile libraries
- Set environment variables 
- In case of different version of CUDA than 9.0, change "cuda_version" variable in premake5.lua to your version.
- Run commmand 
```/code/build/premake5(.exe) [action]```
- ```[action] = vs2017```, for others see: [Using Premake](https://github.com/premake/premake-core/wiki/Using-Premake) 
-






