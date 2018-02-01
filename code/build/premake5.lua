local glfw = os.getenv("GLFW_PATH") -- glfw-3.2.1 
local glew = os.getenv("GLEW_PATH") -- glew-2.0.0
local glm = os.getenv("GLM_PATH") --glm 0.9.8
local eigen = os.getenv("EIGEN_PATH") --eigen 3.3.4

cuda_version = "9.0"
include("premake-cuda/cuda.lua")

--local blib = "../../batterylib" 
local vizDir = "../batteryviz/"
local libDir = "../batterylib/"

workspace "battery"
	configurations { "Debug", "Release" }
	buildoptions { "/openmp /std:c++latest" }
	platforms {"x64"}

filter { "platforms:x64" }
    system "Windows"
    architecture "x64"


project "batterylib"
	kind "SharedLib"
	language "C++"
  	targetdir "../bin/%{cfg.buildcfg}/"

  	files {   	   
      libDir .. "src/**.h", 
      libDir .. "src/**.cpp",      
      libDir .. "src/**.cuh",
      libDir .. "src/**.cu",
      libDir .. "include/**.h",
      libDir .. "external/**.cpp",
      libDir .. "external/**.c",
      libDir .. "external/**.h",
   	}
	
	includedirs {
		libDir .. "src/",		
		libDir .. "src/cuda/",		
		libDir .. "include/",
		libDir .. "external/",		
		glew .. "/include",		
		glm,
		eigen	
	}

	libdirs {
		glew .. "/lib/%{cfg.buildcfg}/%{cfg.platform}/"
	}

	links {
		"opengl32",
		"cudart",
		"cusparse",
		"cusolver"
	} 

	defines {		
		"GLEW_STATIC", 
		"WIN64",
		"BATTERYLIB_EXPORT",
		"_CRT_SECURE_NO_WARNINGS",
		"NOMINMAX"
	}

	filter "configurations:Debug"
    	defines { "DEBUG" }
    	flags { "Symbols" }
    	links {"glew32sd"}        	

	filter "configurations:Release"
		defines { "NDEBUG" }
		optimize "On"		
		links {"glew32s"}


project "batteryviz"
	kind "ConsoleApp"
	language "C++"
  	targetdir "../bin/%{cfg.buildcfg}/"

  	files { 
      vizDir .. "src/**.h", 
      vizDir .. "src/**.cpp",
      vizDir .. "external/imgui/**.cpp",
      vizDir .. "external/imgui/**.h",              
      vizDir .. "external/*.cpp",
      vizDir .. "external/*.h"      
   	}
	
	includedirs {
		"../",--root
		vizDir .. "src/",		
		vizDir .. "external/imgui/",
		vizDir .. "external/",		
		glew .. "/include",		
		glfw .. "/include",
		glm,
		eigen		
	}

	libdirs {
    	glfw .. "/src/%{cfg.buildcfg}/",
    	glew .. "/lib/%{cfg.buildcfg}/%{cfg.platform}/",
    	"../bin/%{cfg.buildcfg}/"
   	}

	links { 	    
	    "glfw3",	    
	    "opengl32",
	    "batterylib"   
	} 

	defines {
		"GLEW_STATIC", 
		"WIN64",
		"NO_IMGUIFILESYSTEM", 
		"GLM_ENABLE_EXPERIMENTAL",
		"_CRT_SECURE_NO_WARNINGS",
		"NOMINMAX"
	}

	filter "configurations:Debug"
    	defines { "DEBUG" }
    	flags { "Symbols" }    
    	links {"glew32sd"}

	filter "configurations:Release"
		defines { "NDEBUG" }
		optimize "On"
		links {"glew32s"}

