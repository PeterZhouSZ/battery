#pragma once

#include "BatteryLibDef.h"

#include <utility>
#include <random>
#include <functional>
#include <mutex>



namespace blib {

	template <typename E, typename D, class ...TArgs >
	struct RandomGenerator {
		RandomGenerator(TArgs && ... args)
			: distribtion(std::forward<TArgs>(args)...)
		{}

		RandomGenerator& operator = (const RandomGenerator &g) {
			engine = g.engine;
			distribtion = g.distribtion;
			return *this;
		}

		RandomGenerator(const RandomGenerator &g) {
			engine = g.engine;
			distribtion = g.distribtion;
		}

		E engine;
		D distribtion;

		typedef typename D::result_type result_type;



		result_type next() {
			std::lock_guard<std::mutex> lock(_mutex);
			return distribtion(engine);
		}
	private:
		std::mutex _mutex;
	};

	using RNGNormal = RandomGenerator<
		std::default_random_engine,
		std::normal_distribution<float>,
		float, float>;
	using RNGUniformInt = RandomGenerator<
		std::default_random_engine,
		std::uniform_int_distribution<int>,
		int, int>;
	using RNGUniformFloat = RandomGenerator<
		std::default_random_engine,
		std::uniform_real_distribution<float>,
		float, float>;



	BLIB_EXPORT void exec(int index, const std::function<void(void)> & f);

	template<typename ... Fargs>
	void exec(int index, const std::function<void(void)> & f, Fargs ... args) {
		if (index == 0)
			f();
		else
			exec(index - 1, args ...);
	}

	template<typename ... Largs>
	void choose(RNGUniformInt & rnd, Largs ... args) {
		int index = rnd.next() % (sizeof...(args));
		exec(index, args ...);
	}




	BLIB_EXPORT int randomBi(RNGUniformInt & rnd);

}