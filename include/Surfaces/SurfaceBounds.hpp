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

/// @class SurfaceBounds
///
/// Interface for surface bounds.
///
/// Surface bounds provide:
/// - inside() checks
/// - the BoundsType and a set of parameters to simplify persistency
///
template <typename Derived, unsigned int ValueSize> class SurfaceBounds {
public:
  ~SurfaceBounds() = default;

  /// Return the bounds type - for persistency optimization
  ///
  /// @return is a BoundsType enum
  ACTS_DEVICE_FUNC BoundsType type() const;

  /// Access method for bound variable store
  ///
  /// @return of the stored values for the boundary object
  ACTS_DEVICE_FUNC ActsVector<ActsScalar, ValueSize> values() const;

  /// Inside check for the bounds object driven by the boundary check directive
  /// Each Bounds has a method inside, which checks if a LocalPosition is inside
  /// the bounds  Inside can be called without/with tolerances.
  ///
  /// @param lposition Local position (assumed to be in right surface frame)
  /// @param bcheck boundary check directive
  /// @return boolean indicator for the success of this operation
  ACTS_DEVICE_FUNC bool inside(const Vector2D &lposition,
                               const BoundaryCheck &bcheck) const;
};

template <typename Derived, unsigned int ValueSize>
ACTS_DEVICE_FUNC inline bool
operator==(const SurfaceBounds<Derived, ValueSize> &lhs,
           const SurfaceBounds<Derived, ValueSize> &rhs) {
  if (&lhs == &rhs) {
    return true;
  }
  return (lhs.type() == rhs.type()) && (lhs.values() == rhs.values());
}

template <typename Derived, unsigned int ValueSize>
ACTS_DEVICE_FUNC inline bool
operator!=(const SurfaceBounds<Derived, ValueSize> &lhs,
           const SurfaceBounds<Derived, ValueSize> &rhs) {
  return !(lhs == rhs);
}

template <typename Derived, unsigned int ValueSize>
inline BoundsType SurfaceBounds<Derived, ValueSize>::type() const {
  return static_cast<Derived &>(*this).type();
}

template <typename Derived, unsigned int ValueSize>
inline ActsVector<ActsScalar, ValueSize>
SurfaceBounds<Derived, ValueSize>::values() const {
  return static_cast<Derived &>(*this).values();
}

template <typename Derived, unsigned int ValueSize>
inline bool
SurfaceBounds<Derived, ValueSize>::inside(const Vector2D &lposition,
                                          const BoundaryCheck &bcheck) const {
  return static_cast<Derived &>(*this).inside(lposition, bcheck);
}

} // namespace Acts
