#include "utility/IOUtility.h"
#include <fstream>
#include <regex>
#include <sstream>




#if defined(__GNUC__)
    #include <experimental/filesystem>
#else
	#include <filesystem>
#endif

namespace fs = std::experimental::filesystem;

using namespace std;

#pragma warning(disable:4996) //localtime

string getBaseDirectory(const string & filepath)
{
	const auto pos = filepath.find_last_of("/\\");
	return filepath.substr(0, pos) + "/";
}

pair<string, string> getBaseAndFile(const string & filepath) {
	const auto pos = filepath.find_last_of("/\\");
	return {
		filepath.substr(0, pos) + "/",
		filepath.substr(pos + 1)
	};
}


string readFileToString(const string & filepath)
{
	string str;

	ifstream f(filepath);
	if (!f.good()) return str;


	f.seekg(0, ios::end);
	str.reserve(f.tellg());
	f.seekg(0, ios::beg);

	str.assign(istreambuf_iterator<char>(f), istreambuf_iterator<char>());

	return str;
}

string readFileWithIncludes(const string & filepath)
{
	const auto path = getBaseAndFile(filepath);
	auto & baseDir = path.first;
	auto & filename = path.second;

	auto str = readFileToString(filepath);

	regex rxInclude("^ *#include +\"(.*)\" *$");

	auto rxBegin = sregex_iterator(str.begin(), str.end(), rxInclude);
	auto rxEnd = sregex_iterator();

	for (auto i = rxBegin; i != rxEnd; ) {
		if (i->size() > 1) {
			auto includeFilename = (*i)[1].str();
			auto includeFilepath = baseDir + "\\" + includeFilename;
			string newSrc = move(readFileToString(includeFilepath));
			str.replace(i->position(), i->length(), newSrc);
			i = sregex_iterator(str.begin(), str.end(), rxInclude);
		}
		else {
			++i;
		}
	}

	return str;
}


vector<string> split(const string & str, char delimiter)
{
	vector<string> res;

	std::istringstream ss(str);
	std::string part;
	while (std::getline(ss, part, delimiter)) {
		res.push_back(std::move(part));
	}

	return res;
}

void streamprintf(std::ostream & os, const char * format)
{
	os << format;
}

std::string timestampString(const std::string & format /*= "%Y_%m_%d_%H_%M_%S"*/)
{
	char buffer[256];
	std::time_t now = std::time(NULL);
	std::tm * ptm = std::localtime(&now);
	std::strftime(buffer, 256, format.c_str(), ptm);
	return std::string(buffer);
}


