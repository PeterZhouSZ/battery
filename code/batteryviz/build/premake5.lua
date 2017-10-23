local glfw = os.getenv("GLFW_PATH") -- glfw-3.2.1 
local glew = os.getenv("GLEW_PATH") -- glew-2.0.0
local glm = os.getenv("GLM_PATH") --glm 0.9.8

solution "batteryviz"
	configurations { "Debug", "Release" }
	buildoptions { "/openmp /std:c++latest" }
	platforms {"x64"}

filter { "platforms:x64" }
    system "Windows"
    architecture "x64"

project "batteryviz"
	kind "ConsoleApp"
	language "C++"
  	targetdir "../bin/%{cfg.buildcfg}/"

  	files { 
      "../src/**.h", 
      "../src/**.cpp",
      "../external/imgui/**.cpp",
      "../external/imgui/**.h",        
      "../external/*.cpp",
      "../external/*.h"      
   	}
	
	includedirs {
		"../src/",		
		"../external/imgui/",
		"../external/",
		glew .. "/include",
		glfw .. "/include",
		glm		
	}

	libdirs {
    	glfw .. "/src/%{cfg.buildcfg}/",
    	glew .. "/lib/%{cfg.buildcfg}/%{cfg.platform}/"    	
   	}

	links { 	    
	    "glfw3",	    
	    "opengl32"	    
	} 

	defines {
		"GLEW_STATIC", 
		"WIN64",
		"NO_IMGUIFILESYSTEM", 
		"GLM_ENABLE_EXPERIMENTAL"
	}

	filter "configurations:Debug"
    	defines { "DEBUG" }
    	flags { "Symbols" }    
    	links {"glew32sd"}

	filter "configurations:Release"
		defines { "NDEBUG" }
		optimize "On"
		links {"glew32s"}

