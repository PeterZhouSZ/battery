#version 440 core

#pragma VERTEX

#include "passthrough.vert"

#pragma FRAGMENT

in vec3 vposition;
uniform sampler2D tex;

out vec4 fragColor;

void main(){

	vec2 coord = (vposition.xy + vec2(1)) * 0.5;

	//sample tex
	//...

	fragColor.xy = coord;
	fragColor.z = 0;
	fragColor.a = 1;

}