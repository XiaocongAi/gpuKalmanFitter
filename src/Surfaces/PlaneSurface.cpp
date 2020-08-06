// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// PlaneSurface.cpp, Acts project
///////////////////////////////////////////////////////////////////

#include "Surfaces/PlaneSurface.hpp"
#include "Surfaces/InfiniteBounds.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

Acts::PlaneSurface::PlaneSurface(const PlaneSurface &other)
    : GeometryObject(), Surface(other), m_bounds(other.m_bounds) {}

Acts::PlaneSurface::PlaneSurface(const GeometryContext &gctx,
                                 const PlaneSurface &other,
                                 const Transform3D &transf)
    : GeometryObject(), Surface(gctx, other, transf), m_bounds(other.m_bounds) {
}

Acts::PlaneSurface::PlaneSurface(const Vector3D &center, const Vector3D &normal)
    //: Surface(), m_bounds(RectangleBounds(std::numeric_limits<double>::max(),
    // std::numeric_limits<double>::max())) {
    : Surface(), m_bounds(nullptr) {
  /// the right-handed coordinate system is defined as
  /// T = normal
  /// U = Z x T if T not parallel to Z otherwise U = X x T
  /// V = T x U
  Vector3D T = normal.normalized();
  Vector3D U = std::abs(T.dot(Vector3D::UnitZ())) < s_curvilinearProjTolerance
                   ? Vector3D::UnitZ().cross(T).normalized()
                   : Vector3D::UnitX().cross(T).normalized();
  Vector3D V = T.cross(U);
  RotationMatrix3D curvilinearRotation;
  curvilinearRotation.col(0) = U;
  curvilinearRotation.col(1) = V;
  curvilinearRotation.col(2) = T;

  // curvilinear surfaces are boundless
  Transform3D transform{curvilinearRotation};
  transform.pretranslate(center);
  Surface::m_transform = transform;
}

Acts::PlaneSurface::PlaneSurface(const Transform3D &htrans,
                                 const PlanarBounds *pbounds)
    : Surface(std::move(htrans)), m_bounds(std::move(pbounds)) {}

Acts::PlaneSurface &Acts::PlaneSurface::operator=(const PlaneSurface &other) {
  if (this != &other) {
    Surface::operator=(other);
    m_bounds = other.m_bounds;
  }
  return *this;
}

Acts::Surface::SurfaceType Acts::PlaneSurface::type() const {
  return Surface::Plane;
}

void Acts::PlaneSurface::localToGlobal(const GeometryContext &gctx,
                                       const Vector2D &lposition,
                                       const Vector3D & /*gmom*/,
                                       Vector3D &position) const {
  Vector3D loc3Dframe(lposition[Acts::eLOC_X], lposition[Acts::eLOC_Y], 0.);
  /// the chance that there is no transform is almost 0, let's apply it
  position = transform(gctx) * loc3Dframe;
}

bool Acts::PlaneSurface::globalToLocal(const GeometryContext &gctx,
                                       const Vector3D &position,
                                       const Vector3D & /*gmom*/,
                                       Acts::Vector2D &lposition) const {
  /// the chance that there is no transform is almost 0, let's apply it
  Vector3D loc3Dframe = (transform(gctx).inverse()) * position;
  lposition = Vector2D(loc3Dframe.x(), loc3Dframe.y());
  return ((loc3Dframe.z() * loc3Dframe.z() >
           s_onSurfaceTolerance * s_onSurfaceTolerance)
              ? false
              : true);
}

std::string Acts::PlaneSurface::name() const { return "Acts::PlaneSurface"; }

const Acts::SurfaceBounds &Acts::PlaneSurface::bounds() const {
  if (m_bounds) {
    return *m_bounds;
  }
  return s_noBounds;
}
