// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// PlanarBounds.h, Acts project
///////////////////////////////////////////////////////////////////

#pragma once

#include "Surfaces/SurfaceBounds.hpp"
namespace Acts {

/// Forward declare rectangle bounds as boundary box
class RectangleBounds;

/// @class PlanarBounds
///
/// common base class for all bounds that are in a local x/y cartesian frame
///  - simply introduced to avoid wrong bound assigments to surfaces
///
template <typename Derived, unsigned int VerticeSize, unsigned int ValueSize>
class PlanarBounds
    : public SurfaceBounds<PlanarBounds<Derived, VerticeSize, ValueSize>,
                           ValueSize> {
public:
  /// Return the vertices - or, the points of the extremas
  ActsMatrix<ActsScalar, VerticeSize, 2> vertices() const;

  // Bounding box parameters
  //  const RectangleBounds &boundingBox() const;

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

template <typename Derived, unsigned int VerticeSize, unsigned int ValueSize>
inline ActsMatrix<ActsScalar, VerticeSize, 2>
PlanarBounds<Derived, VerticeSize, ValueSize>::vertices() const {
  return static_cast<Derived &>(*this).vertices();
}

template <typename Derived, unsigned int VerticeSize, unsigned int ValueSize>
inline BoundsType PlanarBounds<Derived, VerticeSize, ValueSize>::type() const {
  return static_cast<Derived &>(*this).type();
}

template <typename Derived, unsigned int VerticeSize, unsigned int ValueSize>
inline ActsVector<ActsScalar, ValueSize>
PlanarBounds<Derived, VerticeSize, ValueSize>::values() const {
  return static_cast<Derived &>(*this).values();
}

template <typename Derived, unsigned int VerticeSize, unsigned int ValueSize>
inline bool PlanarBounds<Derived, VerticeSize, ValueSize>::inside(
    const Vector2D &lposition, const BoundaryCheck &bcheck) const {
  return static_cast<Derived &>(*this).inside(lposition, bcheck);
}

} // namespace Acts
