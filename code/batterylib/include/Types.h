#pragma once

#include <glm/glm.hpp>

namespace blib {

	using vec2 = glm::vec2;
	using vec3 = glm::vec3;
	using vec4 = glm::vec4;

	using color3 = glm::vec3;
	using color4 = glm::vec4;

	using ivec2 = glm::ivec2;
	using ivec3 = glm::ivec3;
	using ivec4 = glm::ivec4;

	using mat2 = glm::mat2;
	using mat3 = glm::mat3;
	using mat4 = glm::mat4;

	using uint64 = unsigned long long;
	using uint = unsigned int;
	using uchar = unsigned char;

	enum PrimitiveType {
		TYPE_FLOAT = 0,
		TYPE_CHAR,
		TYPE_UCHAR,
		TYPE_INT,
		TYPE_FLOAT3,
		TYPE_FLOAT4
	};	

}