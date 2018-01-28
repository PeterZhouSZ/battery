#pragma once


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


enum Dir {
	X_POS = 0,
	X_NEG = 1,
	Y_POS = 2,
	Y_NEG = 3,
	Z_POS = 4,
	Z_NEG = 5,
	DIR_NONE = 6
};