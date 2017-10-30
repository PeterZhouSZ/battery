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

uniform float whiteOpacity = 0.05;
uniform float blackOpacity = 0.001;

uniform vec3 viewPos;

const vec3 lightDir = vec3(1.0,0.0,1.0);
//const float voxelSize = 1 / 256.0;

uniform vec3 resolution;

vec3 getGradient(vec3 pt){
	return vec3(
		texture(volumeTexture, pt - vec3(resolution.x, 0.0, 0.0)).x - texture(volumeTexture, pt + vec3(resolution.x, 0.0, 0.0)).x,
 		texture(volumeTexture, pt - vec3(0.0, resolution.y, 0.0)).x - texture(volumeTexture, pt + vec3(0.0, resolution.y, 0.0)).x,
 		texture(volumeTexture, pt - vec3(0.0, 0.0, resolution.z)).x - texture(volumeTexture, pt + vec3(0.0, 0.0, resolution.z)).x
 	); 	
}

vec3 getNormal(vec3 pt){
	return normalize(getGradient(pt));
}

float getLightMagnitude(vec3 pos, vec3 n, vec3 view){
	
	float ambient = 0.1;	
	float diff = max(dot(lightDir,n),0);
	float spec = 0.0;
	//if(diff > ambient){
		//vec3 r = reflect(-lightDir,n);
		//spec = max(dot(r,view),0.0);
		//spec = pow(spec,15.0);
	//}

	//return 0.01 * spec + 0.99*diff;
	return diff;

}

void main(){


	//vec2 planePos = vec2(vposition.x+1.0,vposition.y+1.0) * 0.5f;	
	vec2 planePos = (vposition.xy + vec2(1)) * 0.5;
	vec3 enterPt = texture(enterVolumeTex,planePos).xyz;
	vec3 exitPt = texture(exitVolumeTex,planePos).xyz;	
	

	if(enterPt == vec3(0.0f)) {				
		fragColor = vec4(0,0,0,0);
		return;
	}
	
	vec3 ray = -normalize(exitPt-enterPt);
	//float dt = distance(exitPt,enterPt) / steps;
	float dt = 0.005;
	float N = distance(exitPt,enterPt) / dt;
	vec3 stepVec = ray*dt;

	vec3 pos = exitPt;//enterPt;
	fragColor = vec4(vec3(0.0),0.0);


	vec4 colorAcc = vec4(0);
	float alphaAcc = 0;


	
	for(float i=0; i < N; i+=1.0){

		float volumeVal = texture(volumeTexture,pos).r;
		//float volumeVal = sampleKernel(pos);

		pos += stepVec;

		//vec4 color = texture(transferFunc,volumeVal);
		vec4 color = vec4(0,0,1,1);



		if(volumeVal > 0.5){
			color = vec4(0.5,0.5,0.5,whiteOpacity * dt * 1000);
		}
		else{
			color = vec4(0,0,0,blackOpacity * dt * 1000);
		}




		vec3 gradient = getGradient(pos);
		float glen = length(gradient);


		//color.rgb = vec3(getLightMagnitude(pos, getNormal(pos), viewPos));
		//if(glen > 0.01)		
		//	color.rgb *= (getLightMagnitude(pos, getNormal(pos), viewPos));

		//color = vec4(vec3(0), glen*glen*glen*glen * dt);
		
		
 
		//vec3 Cprev = colorAcc.xyz * colorAcc.a;
		//float Aprev = colorAcc.a;
		//vec3 Ci = color.xyz * color.a;
		//float Ai = color.a;

		//colorAcc.xyz =  (1 - Ai) * Ci + Cprev;
		//colorAcc.a = (1 - Ai) *  Ai + Aprev;

		//colorAcc.xyz = (1 - Aprev) * Ci + Cprev;
		//colorAcc.a = (1 - Aprev) * Ai + Aprev;

		//colorAcc = color*10;

		if(true){
			color.rgb *= color.a;
			//float alphaSample = 1.0 - pow((1.0 - color.a),dt);

			colorAcc =  (1 - colorAcc.a) * color +  colorAcc;

			//alphaAcc += alphaSample;
			//colorAcc.rgb  = color.rgb * 100;
			//colorAcc.a = 1;
		}




//		if(alphaAcc > 0.95)
			//break;

		//color= vec4(vec3(volumeVal),0.5);
		//color.rgb *= getLightMagnitude(pos,getNormal(pos),ray);
		//color.a *= 1.0;//transferOpacity;		
		//fragColor.rgb = mix(fragColor.rgb, color.rgb, color.a);

		//vec3 Cprev = fragColor.xyz ;
		//float Aprev = fragColor.a;
		//vec3 Ci = color.xyz ;
		//float Ai = color.a;

		//fragColor.xyz =  (1 - Ai) * Ci + Cprev;
		//fragColor.a = (1 - Ai) *  Ai + Aprev;

		//fragColor.xyz = (1 - Aprev) * Ci + Cprev;
		//fragColor.a = (1 - Aprev) * Ai + Aprev;


		//fragColor.rgb = (1 - color.a) * fragColor.rgb + color.rgb;
		//fragColor.a = (1 - color.a)

		//fragColor.a = mix(color.a,1.0,fragColor.a) ;	
		
		
	}

	vec3 clearColor = vec3(1.0);

	//colorAcc.rgb = mix(clearColor, colorAcc.rgb, colorAcc.a);

	fragColor = colorAcc;
	
}