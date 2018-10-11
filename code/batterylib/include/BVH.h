#pragma once

#include "BatteryLibDef.h"
#include <AABB.h>
#include <vector>
#include <memory>

namespace blib {

	class GeometryObject;

	class BVH {

	public:

		using ObjPtr = std::unique_ptr<GeometryObject>;
		using ObjList = std::vector<std::unique_ptr<GeometryObject>>;
		using ObjIterator = std::vector<std::unique_ptr<GeometryObject>>::iterator;

		struct Node {
			using NodePtr = std::unique_ptr<Node>;
			std::vector<ObjPtr> objects;
			std::vector<NodePtr> children;
			AABB _bounds;
		};

		using NodePtr = std::unique_ptr<Node>;
		NodePtr _root;


		BLIB_EXPORT void build(std::vector<std::unique_ptr<GeometryObject>> & objects);


	private:

		void buildRecursive(NodePtr & node, ObjIterator begin, ObjIterator end);

		AABB boundsOfList(const ObjIterator & begin, const ObjIterator & end);

	};

}