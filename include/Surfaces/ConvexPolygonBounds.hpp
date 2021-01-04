// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Surfaces/PlanarBounds.hpp"
#include "Surfaces/RectangleBounds.hpp"
#include "Utilities/Definitions.hpp"

#include <cmath>
#include <exception>

namespace Acts {

/// This is the actual implementation of the bounds.
/// It is templated on the number of vertices, but there is a specialization for
/// *dynamic* number of vertices, where the underlying storage is then a vector.
///
/// @tparam N Number of vertices
template <int N>
class ConvexPolygonBounds
    : public PlanarBounds<ConvexPolygonBounds<N>, N, N * 2> {
public:
  /// Expose number of vertices given as template parameter.
  ///
  static constexpr size_t num_vertices = N;
  /// Type that's used to store the vertices, in this case a fixed size array.
  ///
  using vertex_array = ActsMatrix<ActsScalar, num_vertices, 2>;
  /// Expose number of parameters as a template parameter
  ///
  static constexpr size_t eSize = 2 * N;
  /// Type that's used to store the vertices, in this case a fixed size array.
  ///
  using value_array = ActsVector<ActsScalar, eSize>;

  static_assert(N >= 3, "ConvexPolygonBounds needs at least 3 sides.");

  /// class must have default constructor
  ConvexPolygonBounds() = default;

  /// Constructor from a fixed size array of vertices.
  /// This will throw if the vertices do not form a convex polygon.
  /// @param vertices The vertices
  ConvexPolygonBounds(const vertex_array &vertices) noexcept(false);

  /// Constructor from a fixed size array of parameters
  /// This will throw if the vertices do not form a convex polygon.
  /// @param values The values to build up the vertices
  ConvexPolygonBounds(const value_array &values) noexcept(false);

  ~ConvexPolygonBounds() = default;

  ACTS_DEVICE_FUNC BoundsType type() const;

  /// Return the bound values as dynamically sized vector
  ///
  /// @return this returns a copy of the internal values
  ACTS_DEVICE_FUNC ActsVector<ActsScalar, eSize> values() const;

  /// Return whether a local 2D point lies inside of the bounds defined by this
  /// object.
  /// @param lposition The local position to check
  /// @param bcheck The `BoundaryCheck` object handling tolerances.
  /// @return Whether the points is inside
  ACTS_DEVICE_FUNC bool inside(const Vector2D &lposition,
                               const BoundaryCheck &bcheck) const;

  /// Return the vertices - or, the points of the extremas
  vertex_array vertices() const;

  /// Return a rectangle bounds object that encloses this polygon.
  /// @return The rectangular bounds
  //  const RectangleBounds& boundingBox() const ;

private:
  vertex_array m_vertices = vertex_array::Zero();
  // RectangleBounds m_boundingBox;

  /// Return whether this bounds class is in fact convex
  /// throws a log error if not
  //  void checkConsistency() const noexcept(false);
};

} // namespace Acts

#include "Surfaces/ConvexPolygonBounds.ipp"
