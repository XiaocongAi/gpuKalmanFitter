// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*
void Acts::ConvexPolygonBoundsBase<Derived, N>::convex_impl(
    const ActsMatrix<double, N, 2>& vertices) noexcept(false) {
  const size_t N = vertices.rows();

  for (size_t i = 0; i < N; i++) {
    size_t j = (i + 1) % N;
    const Vector2D& a = vertices.block<1,2>([i],0).transpose();
    const Vector2D& b = vertices.block<1,2>([j],0).transpose();

    const Vector2D ab = b - a;
    const Vector2D normal = Vector2D(ab.y(), -ab.x()).normalized();

    bool first = true;
    bool ref;
    // loop over all other vertices
    for (size_t k = 0; k < N; k++) {
      if (k == i || k == j) {
        continue;
      }

      const Vector2D& c = vertices.block<1,2>([k],0).transpose();
      double dot = normal.dot(c - a);

      if (first) {
        ref = std::signbit(dot);
        first = false;
        continue;
      }

      if (std::signbit(dot) != ref) {
        throw std::logic_error(
            "ConvexPolygon: Given vertices do not form convex hull");
      }
    }
  }
}
*/

// template <typename coll_t>
// Acts::RectangleBounds Acts::ConvexPolygonBoundsBase::makeBoundingBox(
//    const coll_t& vertices) {
//  Vector2D vmax, vmin;
//  vmax = vertices[0];
//  vmin = vertices[0];
//
//  for (size_t i = 1; i < vertices.size(); i++) {
//    vmax = vmax.cwiseMax(vertices[i]);
//    vmin = vmin.cwiseMin(vertices[i]);
//  }
//
//  return {vmin, vmax};
//}

template <int N>
Acts::ConvexPolygonBounds<N>::ConvexPolygonBounds(
    const vertex_array &vertices) noexcept(false)
    : m_vertices(vertices) {
  // checkConsistency();
}

template <int N>
Acts::ConvexPolygonBounds<N>::ConvexPolygonBounds(
    const value_array &values) noexcept(false) {
  for (size_t i = 0; i < N; i++) {
    m_vertices(i, 0) = values[2 * i];
    m_vertices(i, 1) = values[2 * i + 1];
  }
  // checkConsistency();
}

template <int N> Acts::BoundsType Acts::ConvexPolygonBounds<N>::type() const {
  return BoundsType::ConvexPolygon;
}

template <int N>
ActsVector<double, ConvexPolygonBounds<N>::eSize>
Acts::ConvexPolygonBounds<N>::values() const {
  ActsVector<double, ConvexPolygonBounds<N>::eSize> values =
      ActsVector<double, ConvexPolygonBounds<N>::eSize>::Zero();
  unsigned int iValue = 0;
  for (const auto &vtx : vertices()) {
    values[iValue] = vtx.x();
    values[iValue + 1] = vtx.y();
    iValue += 2;
  }
  return values;
}

template <int N>
bool Acts::ConvexPolygonBounds<N>::inside(
    const Acts::Vector2D &lposition, const Acts::BoundaryCheck &bcheck) const {
  return bcheck.isInside(lposition, m_vertices);
}

template <int N>
typename Acts::ConvexPolygonBounds<N>::vertex_array
Acts::ConvexPolygonBounds<N>::vertices() const {
  return m_vertices;
}

// template <int N>
// const Acts::RectangleBounds& Acts::ConvexPolygonBounds<N>::boundingBox()
// const {
//  return m_boundingBox;
//}

// template <int N>
// void Acts::ConvexPolygonBounds<N>::checkConsistency() const noexcept(false) {
//  convex_impl(m_vertices);
//}
