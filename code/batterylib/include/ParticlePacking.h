#pragma once

#include <vector>


namespace blib {


	//transforms + polyhedrons + spatial acceleration structure

	using Particle = int;
	using Tree = int;
	using Transform = int;

	class ParticlePacking {

		std::vector<Particle> particles;
		std::vector<Transform> transforms;		
		Tree tree;

	};


}