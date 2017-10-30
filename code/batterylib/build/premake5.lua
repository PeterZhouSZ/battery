--local glfw = os.getenv("GLFW_PATH") -- glfw-3.2.1 
--local glew = os.getenv("GLEW_PATH") -- glew-2.0.0
--local glm = os.getenv("GLM_PATH") --glm 0.9.8
local eigen = os.getenv("EIGEN_PATH") --eigen 3.3.4

solution "batterylib"
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
  	  "../include/**.h", 
      "../src/**.h", 
      "../src/**.cpp",      
   	}
	
	includedirs {
		"../src/",		
		"../include/",		
		eigen	
	}

	libdirs {
    	
   	}

	links { 	    
		    
	} 

	defines {		
		"WIN64",
		"BATTERYLIB_EXPORT"
	}

	filter "configurations:Debug"
    	defines { "DEBUG" }
    	flags { "Symbols" }        	

	filter "configurations:Release"
		defines { "NDEBUG" }
		optimize "On"		

