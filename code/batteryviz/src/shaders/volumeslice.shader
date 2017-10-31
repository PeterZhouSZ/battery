#version 440 core

////////////////////////////////////////
#pragma VERTEX

#include "passthrough.vert"


////////////////////////////////////////
#pragma FRAGMENT

in vec3 vposition;
uniform sampler3D tex;

uniform float slice;
//uniform mat3 R;
uniform int axis;

out vec4 fragColor;

void main(){

	vec2 coord = (vposition.xy + vec2(1)) * 0.5;	



	vec3 coord3D;
	if(axis == 0){
		coord.x = 1.0 - coord.x;
		coord3D = vec3(coord, slice);
	}
	else if(axis == 1){
		coord3D = vec3(1.0 - coord.x, slice, 1.0 - coord.y);
	}
	else if(axis == 2)	
		coord3D = vec3(slice, coord.x, coord.y);

	//vec3 coord3D = R * vec3(coord,slice);
	
	float r = texture(tex, coord3D).r;

	fragColor.xyz = vec3(r);		
	fragColor.a = 1;
}