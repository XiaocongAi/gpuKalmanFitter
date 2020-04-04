// This file is part of the Acts project.
//
// Copyright (C) 2017 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Utilities//Definitions.hpp"
#include "Utilities/detail/AxisFwd.hpp"
#include "Utilities/detail/GridFwd.hpp"

#include <memory>
#include <string>
#include <tuple>

// Forward declarations
namespace Acts {
template <typename G> struct InterpolatedBFieldMapper;

template <typename M> class InterpolatedBFieldMap;

// class ConstantBField;
} // namespace Acts

// namespace BField {
//  class ScalableBField;
//}

using InterpolatedMapper2D = Acts::InterpolatedBFieldMapper<Acts::detail::Grid<
    Acts::Vector2D, Acts::ATLASBFieldSize, Acts::detail::EquidistantAxis,
    Acts::detail::EquidistantAxis>>;

using InterpolatedMapper3D = Acts::InterpolatedBFieldMapper<Acts::detail::Grid<
    Acts::Vector3D, Acts::ATLASBFieldSize, Acts::detail::EquidistantAxis,
    Acts::detail::EquidistantAxis, Acts::detail::EquidistantAxis>>;

using InterpolatedBFieldMap2D =
    Acts::InterpolatedBFieldMap<InterpolatedMapper2D>;
using InterpolatedBFieldMap3D =
    Acts::InterpolatedBFieldMap<InterpolatedMapper3D>;

namespace Options {
// create the bfield maps
InterpolatedBFieldMap3D readBField(std::string bfieldmap);
} // namespace Options
