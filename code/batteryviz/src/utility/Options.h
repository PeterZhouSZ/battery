#pragma once

#include "mathtypes.h"




#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
//#define NO_VARIANT

#if defined(NO_VARIANT)

enum OptionType {	
	OT_STRING = 0,
	OT_CHAR,
	OT_INT,
	OT_FLOAT,
	OT_DOUBLE,
	OT_BOOL,
	OT_VEC2,
	OT_VEC3,
	OT_VEC4,
	OT_IVEC2,
	OT_IVEC3,
	OT_IVEC4,
	OT_MAT2,
	OT_MAT3,
	OT_MAT4,
	OT_NONE
};

union OptionValue {
	std::string _string;
	char _char;
	int _int; 
	float _float;
	double _double;
	bool _bool; 
	vec2 _vec2; 
	vec3 _vec3;
	vec4 _vec4;
	ivec2 _ivec2;
	ivec3 _ivec3:
	ivec4 _ivec4;
	mat2 _mat2;
	mat3 _mat3;
	mat4 _mat4;
};

struct Option {
	OptionType type = OT_NONE;
	OptionValue value;
};


#else 

#include <variant>

using OptionType = std::variant<
	std::string,
	char, int, 
	float, double, 
	bool, 
	vec2, vec3,	vec4,
	ivec2, ivec3, ivec4,
	mat2, mat3,	mat4	
>;

struct Option {
	Option(){}
	~Option(){}
	OptionType value;			
};

#endif



struct OptionSet {	
	std::unordered_map<std::string,Option> options;
	std::unordered_map<std::string,std::unique_ptr<OptionSet>> children;

	OptionSet & operator[](const std::string & childName);	
	const OptionSet & operator[](const std::string & childName) const;

	template <typename T>
	T & get(const std::string & optionName);


	template <typename T>
	void set(const std::string & optionName, T value);

	bool erase(const std::string & optionName);

	void clear();

};

template <typename T>
T & OptionSet::get(const std::string & optionName)
{
	auto & opt = options[optionName];

	//Construct if empty
	#if defined(NO_VARIANT)
		T & value = *static_cast<T*>(&opt.value);
		if(opt.type == OT_NONE){
			value = T();
		}

		return value;

	#else
		if (!std::holds_alternative<T>(opt.value))
			opt.value = T();	

		return std::get<T>(options[optionName].value);
	#endif
}

template <typename T>
void OptionSet::set(const std::string & optionName, T value)
{
	get<T>(optionName) = value;
}



std::ostream & operator << (std::ostream &, const std::unique_ptr<OptionSet> & opt);
std::istream & operator >> (std::istream &, std::unique_ptr<OptionSet> & opt);

std::ostream & operator << (std::ostream &, const OptionSet & opt);
std::istream & operator >> (std::istream &, OptionSet & opt);
