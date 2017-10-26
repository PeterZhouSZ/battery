#include "RandomGenerator.h"

void exec(int index, const std::function<void(void)> & f)
{

	f();
}


int randomBi(RNGUniformInt & rnd) {
	return (rnd.next() % 2) * 2 - 1;
}