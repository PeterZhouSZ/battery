#include "Options.h"

//#define NO_VARIANT

#include "json.hpp"
#include "glm/gtc/type_ptr.hpp"

using json = nlohmann::json;
using namespace std;

json toJson(const std::shared_ptr<OptionSet> & optSet);
json toJson(const OptionSet & optSet);

OptionSet & OptionSet::operator[](const std::string & childName)
{
	auto & ptr = children[childName];
	if(!ptr)
		children[childName] = std::make_unique<OptionSet>();

	return *children[childName];
}

const OptionSet & OptionSet::operator[](const std::string & childName) const
{
	return *children.at(childName);
}

bool OptionSet::erase(const std::string & optionName)
{
	auto it = options.find(optionName);
	if (it == options.end()) return false;
	
	options.erase(it);
	return true;
}

void OptionSet::clear()
{
	options.clear();
	children.clear();
}

template <typename T>
json vecJson(const T & v) {
	json j;
	for (auto i = 0; i < v.length(); i++) {
		j.push_back(v[i]);
	}
	return j;
}

template <typename T>
json matJson(const T & v) {
	json j;
	for (auto i = 0; i < v.length() * v.length(); i++) {		
		j.push_back( glm::value_ptr(v)[i] );		
	}
	return j;
}

template <typename T>
json toJson(const T & v) { return json(v); }

json toJson(const vec2 & v) { return vecJson(v); }
json toJson(const vec3 & v) { return vecJson(v); }
json toJson(const vec4 & v) { return vecJson(v); }
json toJson(const ivec2 & v) { return vecJson(v); }
json toJson(const ivec3 & v) { return vecJson(v); }
json toJson(const ivec4 & v) { return vecJson(v); }
json toJson(const mat2 & v) { return matJson(v); }
json toJson(const mat3 & v) { return matJson(v); }
json toJson(const mat4 & v) { return matJson(v); }

json toJson(const std::string &v) {
	return json(v);
}


json toJson(const OptionSet & optSet) {
	
	json j;

	for (auto & it : optSet.children) {
		j[it.first] = toJson(it.second);		
	}

	
	for (auto & it : optSet.options) {
		std::visit(
			[&](auto&& arg) {

				j[it.first] = {
					{ "type", it.second.value.index() },
					{ "data", toJson(arg) }
				};				
			},
			it.second.value
		);
	}
	

	return j;

}

json toJson(const std::shared_ptr<OptionSet> & optSet) {
	return toJson(*optSet);
}

std::ostream & operator<<(std::ostream & os, const std::shared_ptr<OptionSet> & optSet)
{
	os << toJson(optSet).dump(4);
	return os;
}


std::ostream & operator<<(std::ostream & os, const OptionSet & optSet)
{
	//auto x = std::make_shared<OptionSet>(optSet);		
	//os << toJson(x).dump(4);
	//std::cout << "NOT IMPLEMENTED" << std::endl;
	os << toJson(optSet).dump(4);
	return os;
}




template <typename T>
T jsonToVec(const json & j) {
	T v;
	for (auto i = 0; i < v.length(); i++)
		v[i] = j[i];
	return v;		
}

template <typename T>
T jsonToMat(const json & j) {
	T v;
	for (auto i = 0; i < v.length()*v.length(); i++)
		glm::value_ptr(v)[i] = j[i];
	return v;
}

template <typename T>
void fromJson(const json & j, T & v) { v = j.get<T>(); }
void fromJson(const json & j, vec2 & v) { v = jsonToVec<vec2>(j); }
void fromJson(const json & j, vec3 & v) { v = jsonToVec<vec3>(j); }
void fromJson(const json & j, vec4 & v) { v = jsonToVec<vec4>(j); }
void fromJson(const json & j, ivec2 & v) { v = jsonToVec<ivec2>(j); }
void fromJson(const json & j, ivec3 & v) { v = jsonToVec<ivec3>(j); }
void fromJson(const json & j, ivec4 & v) { v = jsonToVec<ivec4>(j); }
void fromJson(const json & j, mat2 & v) { v = jsonToMat<mat2>(j); }
void fromJson(const json & j, mat3 & v) { v = jsonToMat<mat3>(j); }
void fromJson(const json & j, mat4 & v) { v = jsonToMat<mat4>(j); }

#if defined(NO_VARIANT)
template<typename T>
void set(OptionSet &opt, json::const_iterator & it) {	
	T val;
	fromJson((*it)["data"], val);
	opt.set<T>(it.key(), val);
};
#else
template<size_t idx>
void set(OptionSet &opt, json::const_iterator & it) {
	using T = variant_alternative_t<idx, OptionType>;
	T val;
	fromJson((*it)["data"], val);
	opt.set<T>(it.key(), val);
};
#endif


OptionSet fromJson(const json & j) {
	OptionSet opt;

	for (auto it = j.begin(); it != j.end(); ++it) {		
		if ((*it).find("type") == (*it).end()) {
			opt[it.key()] = fromJson(*it);			
		}
		else {	
			#if defined(NO_VARIANT)
			switch((*it)["type"].get<int>()){
				case OT_STRING:
					set<std::string>(opt, it);
					break;
				case OT_CHAR:
					set<char>(opt, it);
					break;
				case OT_INT:
					set<int>(opt, it);
					break;
				case OT_FLOAT:
					set<float>(opt, it);
					break;
				case OT_DOUBLE:
					set<double>(opt, it);
					break;
				case OT_BOOL:
					set<bool>(opt, it);
					break;
				case OT_VEC2:
					set<vec2>(opt, it);
					break;
				case OT_VEC3:
					set<vec3>(opt, it);
					break;
				case OT_VEC4:
					set<vec4>(opt, it);
					break;
				case OT_IVEC2:
					set<ivec2>(opt, it);
					break;
				case OT_IVEC3:
					set<ivec3>(opt, it);
					break;
				case OT_IVEC4:
					set<ivec4>(opt, it);
					break;
				case OT_MAT2:
					set<mat2>(opt, it);
					break;
				case OT_MAT3:
					set<mat3>(opt, it);
					break;
				case OT_MAT4:
					set<mat4>(opt, it);
					break;				
			};


			#else
			switch ((*it)["type"].get<int>() ) {
				case 0: set<0>(opt, it); break;	case 1: set<1>(opt, it); break;
				case 2: set<2>(opt, it); break;	case 3: set<3>(opt, it); break;
				case 4: set<4>(opt, it); break;	case 5: set<5>(opt, it); break;
				case 6: set<6>(opt, it); break; case 7: set<7>(opt, it); break;
				case 8: set<8>(opt, it); break; case 9: set<9>(opt, it); break;
				case 10: set<10>(opt, it); break; case 11: set<11>(opt, it); break;
				case 12: set<12>(opt, it); break; case 13: set<13>(opt, it); break;
				case 14: set<14>(opt, it); break;
			};			
			#endif
		}
	}

	return opt;

}


std::istream & operator>>(std::istream & is, std::shared_ptr<OptionSet> & opt)
{
	json j;
	is >> j;
	
	opt = std::make_unique<OptionSet>(fromJson(j));

	return is;
}

std::istream & operator>>(std::istream & is, OptionSet & opt)
{
	json j;
	is >> j;
	
	opt = fromJson(j);

	return is;
}