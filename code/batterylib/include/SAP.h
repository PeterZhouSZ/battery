#pragma once

#include "BatteryLibDef.h"
#include <memory>
#include <vector>

namespace blib {

	struct GeometryObject;

	class SAP {
		
		using ObjPtr = std::shared_ptr<GeometryObject>;
		using ObjArr = std::vector<ObjPtr>;
		using CollisionPair = std::pair<ObjPtr, ObjPtr>;
		

	public:
		BLIB_EXPORT void build(const std::vector<std::shared_ptr<GeometryObject>> & objects, int axis = 0);
		
		BLIB_EXPORT std::vector<CollisionPair> getCollisionPairs() const;
		
	private:
		using CollisionPairIndices = std::pair<size_t, size_t>;
		std::vector<CollisionPairIndices> _collisions;
		ObjArr _array;

		void sweep();

	};

}