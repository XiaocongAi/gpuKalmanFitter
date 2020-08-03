// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// SurfaceBounds.h, Acts project
///////////////////////////////////////////////////////////////////

#pragma once
#include <ostream>

#include "Surfaces/BoundaryCheck.hpp"
#include "Utilities/Definitions.hpp"

namespace Acts {

/// @class SurfaceBounds
///
/// Interface for surface bounds.
///
/// Surface bounds provide:
/// - inside() checks
/// - distance to boundary calculations
/// - the BoundsType and a set of parameters to simplify persistency
///
class SurfaceBounds {
 public:
  /// @enum BoundsType
  ///
  /// This enumerator simplifies the persistency,
  /// by saving a dynamic_cast to happen.
  ///
  enum BoundsType {
    Cone = 0,
    Cylinder = 1,
    Diamond = 2,
    Disc = 3,
    Ellipse = 5,
    Line = 6,
    Rectangle = 7,
    RotatedTrapezoid = 8,
    Trapezoid = 9,
    Triangle = 10,
    DiscTrapezoidal = 11,
    ConvexPolygon = 12,
    Annulus = 13,
    Boundless = 14,
    Other = 15
  };

  virtual ~SurfaceBounds() = default;

  /// Return the bounds type - for persistency optimization
  ///
  /// @return is a BoundsType enum
  virtual BoundsType type() const = 0;

  /// Access method for bound variable store
  ///
  /// @return of the stored values for the boundary object
  virtual std::vector<TDD_real_t> valueStore() const = 0;

  /// Inside check for the bounds object driven by the boundary check directive
  /// Each Bounds has a method inside, which checks if a LocalPosition is inside
  /// the bounds  Inside can be called without/with tolerances.
  ///
  /// @param lposition Local position (assumed to be in right surface frame)
  /// @param bcheck boundary check directive
  /// @return boolean indicator for the success of this operation
  virtual bool inside(const Vector2D& lposition,
                      const BoundaryCheck& bcheck) const = 0;

  /// Minimal distance to boundary ( > 0 if outside and <=0 if inside)
  ///
  /// @param lposition is the local position to check for the distance
  /// @return is a signed distance parameter
  virtual double distanceToBoundary(const Vector2D& lposition) const = 0;

};

inline bool operator==(const SurfaceBounds& lhs, const SurfaceBounds& rhs) {
  if (&lhs == &rhs) {
    return true;
  }
  return (lhs.type() == rhs.type()) && (lhs.valueStore() == rhs.valueStore());
}

inline bool operator!=(const SurfaceBounds& lhs, const SurfaceBounds& rhs) {
  return !(lhs == rhs);
}

}  // namespace Acts
