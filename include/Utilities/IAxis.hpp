// This file is part of the Acts project.
//
// Copyright (C) 2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Utilities/Definitions.hpp"
#include "Utilities/detail/AxisFwd.hpp"

namespace Acts {

/// Common base class for all Axis instance. This allows generice handling
/// such as for inspection.
class IAxis {
public:
  /// @brief returns whether the axis is equidistant
  ///
  /// @return bool is equidistant
  ACTS_DEVICE_FUNC virtual bool isEquidistant() const = 0;

  /// @brief returns whether the axis is variable
  ///
  /// @return bool is variable
  ACTS_DEVICE_FUNC virtual bool isVariable() const = 0;

  /// @brief returns the boundary type set in the template param
  ///
  /// @return @c AxisBoundaryType of this axis
  ACTS_DEVICE_FUNC virtual detail::AxisBoundaryType getBoundaryType() const = 0;

  /// @brief Return a vector of bin edges
  /// @return Vector which contains the bin edges
  ACTS_DEVICE_FUNC virtual ActsVectorXd getBinEdges() const = 0;

  /// @brief get minimum of binning range
  ///
  /// @return minimum of binning range
  ACTS_DEVICE_FUNC virtual double getMin() const = 0;

  /// @brief get maximum of binning range
  ///
  /// @return maximum of binning range
  ACTS_DEVICE_FUNC virtual double getMax() const = 0;

  /// @brief get total number of bins
  ///
  /// @return total number of bins (excluding under-/overflow bins)
  ACTS_DEVICE_FUNC virtual size_t getNBins() const = 0;
};
} // namespace Acts
