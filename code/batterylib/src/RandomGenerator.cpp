#include "RandomGenerator.h"

using namespace blib;

void blib::exec(int index, const std::function<void(void)> & f)
{
	f();
}


int blib::randomBi(RNGUniformInt & rnd) {
	return (rnd.next() % 2) * 2 - 1;
}