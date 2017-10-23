#version 440 core

////////////////////////////////////////
#pragma VERTEX



in vec3 position;
out vec3 vposition;

uniform mat4 PVM;

void main()
{	   
    gl_Position = PVM * vec4(position,1.0);
    vposition = (position + vec3(1.0))*0.5;
}



////////////////////////////////////////
#pragma FRAGMENT

in vec3 vposition;
out vec4 fragColor;

void main(){
	fragColor.xyz = vposition;		
	fragColor.a = 1;
}