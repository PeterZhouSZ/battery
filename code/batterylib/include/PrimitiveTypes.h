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
	TYPE_FLOAT4,	
	TYPE_DOUBLE,
};

inline uint primitiveSizeof(PrimitiveType type) {
	switch (type) {
	case TYPE_FLOAT: 
		return sizeof(float);
	case TYPE_DOUBLE:
		return sizeof(double);
	case TYPE_CHAR:
		return sizeof(char);
	case TYPE_UCHAR:
		return sizeof(uchar);
	case TYPE_INT:
		return sizeof(int);
	case TYPE_FLOAT3:
		return sizeof(float) * 3;
	case TYPE_FLOAT4:
		return sizeof(float) * 4;
	}
	return 0;
}

inline float primitiveToFloat(PrimitiveType type, void * ptr) {
	switch (type) {
	case TYPE_FLOAT:
		return *((float*)ptr);
	case TYPE_DOUBLE:
		return float(*((double*)ptr));
	case TYPE_CHAR:
		return float(*((char*)ptr));
	case TYPE_UCHAR:
		return float(*((uchar*)ptr));;
	case TYPE_INT:
		return float(*((int*)ptr));;		
	case TYPE_FLOAT3:
		return nan("");
	case TYPE_FLOAT4:
		return nan("");
	}
	return nan("");
}

inline double primitiveToDouble(PrimitiveType type, void * ptr) {
	switch (type) {
	case TYPE_FLOAT:
		return double(*((float*)ptr));
	case TYPE_DOUBLE:
		return (*((double*)ptr));
	case TYPE_CHAR:
		return double(*((char*)ptr));
	case TYPE_UCHAR:
		return double(*((uchar*)ptr));;
	case TYPE_INT:
		return double(*((int*)ptr));;
	case TYPE_FLOAT3:
		return nan("");
	case TYPE_FLOAT4:
		return nan("");
	}

	return nan("");
}



enum Dir {
	X_POS = 0,
	X_NEG = 1,
	Y_POS = 2,
	Y_NEG = 3,
	Z_POS = 4,
	Z_NEG = 5,
	DIR_NONE = 6
};

inline uint getDirIndex(Dir dir) {
	switch (dir) {
		case X_POS:
		case X_NEG:
			return 0;
		case Y_POS:
		case Y_NEG:
			return 1;
		case Z_POS:
		case Z_NEG:
			return 2;
		default:
			return uint(-1);
	}
	return uint(-1);
	
}

inline int getDirSgn(Dir dir) {
	return -((dir % 2) * 2 - 1);
}

inline Dir getDir(int index, int sgn) {
	sgn = (sgn + 1) / 2; // 0 neg, 1 pos
	sgn = 1 - sgn; // 1 neg, 0 pos
	return Dir(index * 2 + sgn);
}


