#version 440 core

////////////////////////////////////////
#pragma VERTEX

#include "passthrough.vert"


////////////////////////////////////////
#pragma FRAGMENT

in vec3 vposition;
out vec4 fragColor;

uniform sampler1D transferFunc;
uniform sampler3D volumeTexture;
uniform sampler2D enterVolumeTex;
uniform sampler2D exitVolumeTex;

uniform int steps;
uniform float transferOpacity;

uniform vec3 minCrop;
uniform vec3 maxCrop;

const vec3 lightDir = vec3(1.0,0.0,1.0);
const float voxelSize = 0.01;

vec3 getNormal(vec3 pt){
	vec3 n = vec3(
		texture(volumeTexture, pt - vec3(voxelSize, 0.0, 0.0)).x - texture(volumeTexture, pt + vec3(voxelSize, 0.0, 0.0)).x,
 		texture(volumeTexture, pt - vec3(0.0, voxelSize, 0.0)).x - texture(volumeTexture, pt + vec3(0.0, voxelSize, 0.0)).x,
 		texture(volumeTexture, pt - vec3(0.0, 0.0, voxelSize)).x - texture(volumeTexture, pt + vec3(0.0, 0.0, voxelSize)).x
 	);

 	return normalize(n);
}

float getLightMagnitude(vec3 pos, vec3 n, vec3 view){
	
	float ambient = 0.1;	
	float diff = max(dot(lightDir,n),ambient);
	float spec = 0.0;
	if(diff > ambient){
		vec3 r = reflect(-lightDir,n);
		spec = max(dot(r,view),0.0);
		spec = pow(spec,15.0);
	}

	return 0.01 * spec + 0.99*diff;

}

void main(){

	vec2 planePos = vec2(vposition.x+1.0,vposition.y+1.0) * 0.5f;	
	vec3 enterPt = texture(enterVolumeTex,planePos).xyz;
	vec3 exitPt = texture(exitVolumeTex,planePos).xyz;	

	if(enterPt == vec3(0.0f)) {		
		return;
	}
	
	vec3 ray = normalize(exitPt-enterPt);
	float dt = distance(exitPt,enterPt) / steps;
	vec3 stepVec = ray*dt;

	vec3 pos = enterPt;
	fragColor = vec4(vec3(0.0),1.0);
	
	for(int i=0; i < steps; i++){

		float volumeVal = texture(volumeTexture,pos);
		vec4 color = texture(transferFunc,volumeVal);
		color.rgb *= getLightMagnitude(pos,getNormal(pos),ray);
		color.a *= transferOpacity;		
		fragColor.rgb = mix(fragColor.rgb, color.rgb, color.a);
		fragColor.a = mix(color.a,1.0,fragColor.a) ;	
		
		pos += stepVec;
	}
	
}