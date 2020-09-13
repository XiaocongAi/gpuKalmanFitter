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

class InfiniteBounds : public SurfaceBounds<InfiniteBounds, 0> {
public:
  InfiniteBounds() = default;
  ~InfiniteBounds() = default;

  ACTS_DEVICE_FUNC BoundsType type() const { return BoundsType::Boundless; }

  ACTS_DEVICE_FUNC ActsVector<double, 0> values() const { return ActsVector<double, 0>::Zero(); }

  /// Method inside() returns true for any case
  ///
  /// ignores input parameters
  ///
  /// @return always true
  ACTS_DEVICE_FUNC bool inside(const Vector2D & /*lposition*/,
                               const BoundaryCheck & /*bcheck*/) const {
    return true;
  }
};

static const InfiniteBounds s_noBounds{};

} // namespace Acts
