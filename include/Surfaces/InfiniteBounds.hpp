// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// InfiniteBounds.h, Acts project
///////////////////////////////////////////////////////////////////

#pragma once
#include "Surfaces/SurfaceBounds.hpp"
#include "Utilities/Definitions.hpp"

namespace Acts {

/// @class InfiniteBounds
///
/// templated boundless extension to forward the interface
/// Returns all inside checks to true and can templated for all bounds

class InfiniteBounds : public SurfaceBounds {
public:
  InfiniteBounds() = default;
  ~InfiniteBounds() override = default;

  SurfaceBounds::BoundsType type() const final {
    return SurfaceBounds::Boundless;
  }

  std::vector<TDD_real_t> valueStore() const final { return {}; }

  /// Method inside() returns true for any case
  ///
  /// ignores input parameters
  ///
  /// @return always true
  bool inside(const Vector2D & /*lposition*/,
              const BoundaryCheck & /*bcheck*/) const final {
    return true;
  }

  /// Minimal distance calculation
  /// ignores input parameter
  /// @return always 0. (should be -NaN)
  double distanceToBoundary(const Vector2D & /*position*/) const final {
    return 0;
  }
};

static const InfiniteBounds s_noBounds{};

} // namespace Acts
