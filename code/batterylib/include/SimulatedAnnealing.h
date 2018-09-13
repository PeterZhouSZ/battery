#pragma once


#include <functional>
#include <limits>

#include "BatteryLibDef.h"
#include "RandomGenerator.h"

namespace blib {
	

	BLIB_EXPORT float defaultAcceptance(float e0, float e1, float temp);
	BLIB_EXPORT float temperatureLinear(float fraction);
	BLIB_EXPORT float temperatureQuadratic(float fraction);
	BLIB_EXPORT float temperatureExp(float fraction);
	
	
	
	template <typename T>
	struct SimulatedAnnealing {

		using value_type = T;

		std::function<float(const T &)> score;
		std::function<T(const T &)> getNeighbour;

		std::function<float(float fraction)> getTemperature = temperatureLinear;
		std::function<float(float state, float newState, float temp)> acceptance = defaultAcceptance;		

		T initState;
		T state;
		float currentScore;
		size_t maxSteps;
		size_t currentStep;

		float lastAcceptanceP;

		T bestState;
		float bestStateScore;

		int sampleScoreCount = 1;

		float currentTemperature() const {
			return getTemperature(currentStep / static_cast<float>(maxSteps));
		}


		void init(const T & initialState, size_t maximumSteps) {
			initState = initialState;
			state = initialState;
			maxSteps = maximumSteps;
			currentStep = 0;
			currentScore = score(initialState);
			lastAcceptanceP = 0.0f;

			bestStateScore = currentScore;
			bestState = state;


		}

		bool update(size_t steps) {

			auto limit = std::min(currentStep + steps, maxSteps);
			//Advance steps times
			for (; currentStep < limit; currentStep++) {

				//Generate new state
				T newState = getNeighbour(state);

				//Score the new state
				float newScore = 0.0f;
				for (auto i = 0; i < sampleScoreCount; i++) {
					newScore += score(newState);
				}
				newScore *= (1.0f / sampleScoreCount);

				//Store best state
				if (newScore < bestStateScore) {
					bestState = newState;
					bestStateScore = newScore;
				}

				//Decide whether to jump
				float temperature = currentTemperature();
				float P = acceptance(currentScore, newScore, temperature);
				lastAcceptanceP = P;
			
				if (newScore != std::numeric_limits<float>::max() && P >= randomUniform()) {
					state = std::move(newState);
					currentScore = newScore;
				}
			}

			if (currentStep == maxSteps) {
				state = bestState;
				currentScore = bestStateScore;
				return false;
			}
			return true;
		}

	private:
		RNGUniformFloat _uniformDistr = RNGUniformFloat(0, 1);

		float randomUniform() {
			return _uniformDistr.next();
		}

	};

}


/*

Templated, compile-time version (not finished)

*/
//SimulatedAnnealing<T>(2048, scoreFunc(), tempFunc(), acceptFunc());

/*template <typename T>
using SAScoreFun = float(const T &);

using SATempFun = float(float);
using SAAcceptFun = float(float, float, float);

template <typename T, SAScoreFun scoreFun, SATempFun temperatureFun, SAAcceptFun acceptFun>
struct SA {

SA(T && initialState, size_t maxSteps) :
_currentState(std::move(initialState)),
_maxSteps(maxSteps),
_currentStep(0),
_currentScore
{

}

float temperature() const {
return temperatureFun(_currentStep / static_cast<float>(_maxSteps));
}

private:
T _currentState;
T _currentScore;
const size_t _maxSteps;
size_t _currentStep;
};*/


