#include "SimulatedAnnealing.h"


float blib::defaultAcceptance(float e0, float e1, float temp)
{
	if (e1 < e0)
		return 1.0f;

	return exp(-(e1 - e0) / temp);
}

float blib::temperatureLinear(float fraction)
{
	return (1.0f - fraction);
}

float blib::temperatureQuadratic(float fraction)
{
	return (-fraction * fraction + 1);
}

float blib::temperatureExp(float fraction)
{
	return exp(-fraction);
}

/*
float blib::defaultTemperature(float fraction) {
	return 1.0f - fraction;
}
*/
