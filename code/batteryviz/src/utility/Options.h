#pragma once

#include "mathtypes.h"

#include <variant>
#include <any>

#include <string>
#include <vector>
#include <unordered_map>


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
	OptionType value;			
};



struct OptionSet {	
	std::unordered_map<std::string,Option> options;
	std::unordered_map<std::string,OptionSet> children;

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
	if (!std::holds_alternative<T>(opt.value))
		opt.value = T();

	return std::get<T>(options[optionName].value);
}

template <typename T>
void OptionSet::set(const std::string & optionName, T value)
{
	options[optionName].value = std::move(value);
}



std::ostream & operator << (std::ostream &, const OptionSet & opt);
std::istream & operator >> (std::istream &, OptionSet & opt);
