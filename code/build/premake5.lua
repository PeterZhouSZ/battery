local glfw = os.getenv("GLFW_PATH") -- glfw-3.2.1 
local glew = os.getenv("GLEW_PATH") -- glew-2.0.0
local glm = os.getenv("GLM_PATH") --glm 0.9.8
local eigen = os.getenv("EIGEN_PATH") --eigen 3.3.4

local cuda = os.getenv("CUDA_PATH")


cuda_version = "9.2"
include("premake-cuda/cuda.lua")

--local blib = "../../batterylib" 
local vizDir = "../batteryviz/"
local libDir = "../batterylib/"
local toolDir = "../batterytool/"

local GLFWLIB = "glfw3"

workspace "battery"
	configurations { "Debug", "Release" }	
	
	architecture "x64"

filter "action:vs*"
	buildoptions { "/openmp /std:c++latest" }
	local OpenGLLib = "opengl32"
	local GLEWLib = "glew32s"
	

filter "action:gmake*"
	buildoptions {"-fopenmp","-w","-std=c++17", "-lstdc++fs"}	
	linkoptions {"-lstdc++fs","-lglfw","-lgomp"}
	local OpenGLLib = "GL"
	local GLEWLib = "GLEW"
	
	


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
		eigen,
		cuda .. "/include"
	}

	filter "action:vs*"
		libdirs {			
			glew .. "/lib/%{cfg.buildcfg}/%{cfg.platform}/",				
		}
	filter "action:gmake*"
		libdirs {
			cuda .. "/lib64/"
		}

	links {
		OpenGLLib,
		GLEWLib,
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
 

	filter "configurations:Release"
		defines { "NDEBUG" }
		optimize "On"
		


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
		libDir .. "/include",
		vizDir .. "src/",		
		vizDir .. "external/imgui/",
		vizDir .. "external/",		
		glew .. "/include",		
		glfw .. "/include",
		glm,
		eigen,
		cuda .. "/include"	
	}


	filter "action:vs*"
		libdirs {
			glfw .. "/src/%{cfg.buildcfg}/",
			glew .. "/lib/%{cfg.buildcfg}/%{cfg.platform}/",
			"../bin/%{cfg.buildcfg}/"
		}

		links { 	    
			GLFWLib,	    
			OpenGLLib,
			GLEWLib,
			"batterylib"   
		} 
	filter "action:gmake*"			
		links { 	    
			GLFWLib,	    
			OpenGLLib,
			GLEWLib,
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

	filter "configurations:Release"
		defines { "NDEBUG" }
		optimize "On"		

project "batterytool"
	kind "ConsoleApp"
	language "C++"
  	targetdir "../bin/%{cfg.buildcfg}/"

  	files { 
      toolDir .. "src/**.h", 
      toolDir .. "src/**.cpp",
      toolDir .. "external/*.cpp",
      toolDir .. "external/*.h"         
   	}

   	includedirs {
		"../",--root
		toolDir .. "src/",				 	
      	toolDir .. "external/",	
--		glew .. "/include",		
--		glfw .. "/include",
		glm,
		eigen		
	}

	libdirs {
		glfw .. "/src/%{cfg.buildcfg}/",
		"../bin/%{cfg.buildcfg}/",
		glew .. "/lib/%{cfg.buildcfg}/%{cfg.platform}/"
	}

	links { 	   
	 	--GLFWLib,	 	        
	    --OpenGLLib,
	    "batterylib"   
	} 

	defines {
		--"GLEW_STATIC", 
		"WIN64",		
		"GLM_ENABLE_EXPERIMENTAL",
		"_CRT_SECURE_NO_WARNINGS",
		"NOMINMAX"
	}

	filter "configurations:Debug"
    	defines { "DEBUG" }
    	flags { "Symbols" }    
--    	links {"glew32sd"}

	filter "configurations:Release"
		defines { "NDEBUG" }
		optimize "On"
--		links {"glew32s"}