#pragma once
#include "BatteryLibDef.h"
#include "AABB.h"
#include "Transform.h"

#include <memory>

namespace blib {


	struct Geometry {

		BLIB_EXPORT virtual AABB bounds() const = 0;
		BLIB_EXPORT virtual std::unique_ptr<Geometry> normalized(bool keepAspectRatio) const = 0;
		BLIB_EXPORT virtual std::unique_ptr<Geometry> transformed(const Transform & t) const = 0;
	};

}