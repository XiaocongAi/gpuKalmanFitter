// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <tuple>
#include <utility>

#include "Plugins/BFieldOptions.hpp"
#include "Plugins/BFieldUtils.hpp"
//#include "MagneticField/ConstantBField.hpp"
#include "Utilities/Units.hpp"

using InterpolatedMapper2D = Acts::InterpolatedBFieldMapper<
    Acts::detail::Grid<Acts::Vector2D, Acts::detail::EquidistantAxis,
                       Acts::detail::EquidistantAxis>>;

using InterpolatedMapper3D = Acts::InterpolatedBFieldMapper<Acts::detail::Grid<
    Acts::Vector3D, Acts::detail::EquidistantAxis,
    Acts::detail::EquidistantAxis, Acts::detail::EquidistantAxis>>;

using InterpolatedBFieldMap2D =
    Acts::InterpolatedBFieldMap<InterpolatedMapper2D>;
using InterpolatedBFieldMap3D =
    Acts::InterpolatedBFieldMap<InterpolatedMapper3D>;

namespace Options {
// create the bfield maps
InterpolatedBFieldMap3D readBField(std::string bfieldmap) {
  enum BFieldMapType { constant = 0, root = 1, text = 2 };

  int bfieldmaptype = text;
  ActsScalar lscalor = 1.;
  ActsScalar bscalor = 1.;

  // Declare the mapper
  ActsScalar lengthUnit = lscalor * Acts::units::_mm;
  ActsScalar BFieldUnit = bscalor * Acts::units::_T;

  // set the mapper - foort
  if (bfieldmaptype == text) {
    auto mapper3D = BField::txt::fieldMapperXYZ(
        [](std::array<size_t, 3> binsXYZ, std::array<size_t, 3> nBinsXYZ) {
          return (binsXYZ.at(0) * (nBinsXYZ.at(1) * nBinsXYZ.at(2)) +
                  binsXYZ.at(1) * nBinsXYZ.at(2) + binsXYZ.at(2));
        },
        bfieldmap, lengthUnit, BFieldUnit, 100, false);

    // create field mapping
    InterpolatedBFieldMap3D::Config config3D(std::move(mapper3D));
    config3D.scale = bscalor;
    // create BField
    InterpolatedBFieldMap3D interpolatedBField3D =
        InterpolatedBFieldMap3D(std::move(config3D));
    return interpolatedBField3D;
  }
}
} // namespace Options
