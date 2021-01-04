// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// RectangleBounds.h, Acts project
///////////////////////////////////////////////////////////////////

#pragma once
#include "Surfaces/SurfaceBounds.hpp"
#include "Utilities/Definitions.hpp"

namespace Acts {

/// @class RectangleBounds
///
/// Bounds for a rectangular, planar surface.
/// The two local coordinates Acts::eLOC_X, Acts::eLOC_Y are for legacy reasons
/// also called @f$ phi @f$ respectively @f$ eta @f$. The orientation
/// with respect to the local surface framce can be seen in the attached
/// illustration.
///
/// @image html RectangularBounds.gif

class RectangleBounds : public PlanarBounds<RectangleBounds, 4, 4> {
public:
  RectangleBounds() = delete;

  /// Constructor with halflength in x and y
  ///
  /// @param halex halflength in X
  /// @param haley halflength in Y
  ACTS_DEVICE_FUNC RectangleBounds(ActsScalar halex, ActsScalar haley);

  /// Constructor with explicit min and max vertex
  ///
  /// @param vmin Minimum vertex
  /// @param vmax Maximum vertex
  ACTS_DEVICE_FUNC RectangleBounds(const Vector2D &vmin, const Vector2D &vmax);

  ~RectangleBounds() = default;

  ACTS_DEVICE_FUNC BoundsType type() const;

  ActsVector<ActsScalar, 4> values() const;

  /// Inside check for the bounds object driven by the boundary check directive
  /// Each Bounds has a method inside, which checks if a LocalPosition is inside
  /// the bounds  Inside can be called without/with tolerances.
  ///
  /// @param lposition Local position (assumed to be in right surface frame)
  /// @param bcheck boundary check directive
  /// @return boolean indicator for the success of this operation
  ACTS_DEVICE_FUNC bool inside(const Vector2D &lposition,
                               const BoundaryCheck &bcheck) const;

  /// Return the vertices - or, the points of the extremas
  ActsMatrix<ActsScalar, 4, 2> vertices() const;

  // Bounding box representation
  //  ACTS_DEVICE_FUNC const RectangleBounds &boundingBox() const;

  /// Return method for the half length in X
  ACTS_DEVICE_FUNC ActsScalar halflengthX() const;

  /// Return method for the half length in Y
  ACTS_DEVICE_FUNC ActsScalar halflengthY() const;

  /// Get the min vertex defining the bounds
  /// @return The min vertex
  ACTS_DEVICE_FUNC const Vector2D &min() const;

  /// Get the max vertex defining the bounds
  /// @return The max vertex
  ACTS_DEVICE_FUNC const Vector2D &max() const;

private:
  Vector2D m_min;
  Vector2D m_max;
};

inline ActsScalar RectangleBounds::halflengthX() const {
  return std::abs(m_max.x() - m_min.x()) * 0.5;
}

inline ActsScalar RectangleBounds::halflengthY() const {
  return std::abs(m_max.y() - m_min.y()) * 0.5;
}

inline BoundsType RectangleBounds::type() const {
  return BoundsType::Rectangle;
}

inline const Vector2D &RectangleBounds::min() const { return m_min; }

inline const Vector2D &RectangleBounds::max() const { return m_max; }

// The following definitions are initially in the cpp file
inline RectangleBounds::RectangleBounds(ActsScalar halex, ActsScalar haley)
    : m_min(-halex, -haley), m_max(halex, haley) {}

inline RectangleBounds::RectangleBounds(const Vector2D &vmin,
                                        const Vector2D &vmax)
    : m_min(vmin), m_max(vmax) {}

inline ActsVector<ActsScalar, 4> RectangleBounds::values() const {
  ActsVector<ActsScalar, 4> values;
  values << m_min.x(), m_min.y(), m_max.x(), m_max.y();
  return values;
}

inline bool RectangleBounds::inside(const Vector2D &lposition,
                                    const BoundaryCheck &bcheck) const {
  return bcheck.isInside(lposition, m_min, m_max);
}

inline ActsMatrix<ActsScalar, 4, 2> RectangleBounds::vertices() const {
  // counter-clockwise starting from bottom-right corner
  ActsMatrix<ActsScalar, 4, 2> vertices = ActsMatrix<ActsScalar, 4, 2>::Zero();
  vertices << m_min.x(), m_min.y(), m_max.x(), m_min.y(), m_max.x(), m_max.y(),
      m_min.x(), m_max.y();
  return vertices;
}

// inline const RectangleBounds &RectangleBounds::boundingBox() const {
//  return (*this);
//}

} // namespace Acts
