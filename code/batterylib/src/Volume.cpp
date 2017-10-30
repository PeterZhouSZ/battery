#include "Volume.h"


Volume<unsigned char> emptyVolume(int size)
{
	Volume<unsigned char> v;
	v.resize(size, size, size);
	return v;
}

