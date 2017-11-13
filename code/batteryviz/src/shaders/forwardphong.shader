#version 430 

#pragma VERTEX
#include "default.vertex"


#pragma FRAGMENT
layout(location = 0) out vec4 fragColor;

in VARYING {
	vec3 pos;
	vec3 normal;
	vec2 uv;
	vec4 color;
} fs_in;


void main(){

	fragColor = vec4(fs_in.color);
}
