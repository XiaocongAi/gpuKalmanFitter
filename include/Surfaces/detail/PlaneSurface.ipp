// This file is part of the Acts project.
//
// Copyright (C) 2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// PlaneSurface.ipp, Acts project
///////////////////////////////////////////////////////////////////

template <typename surface_bounds_t>
inline PlaneSurface<surface_bounds_t>::PlaneSurface(
    const PlaneSurface<surface_bounds_t> &other)
    : GeometryObject(), Surface(other), m_bounds(other.m_bounds) {}

template <typename surface_bounds_t>
inline PlaneSurface<surface_bounds_t>::PlaneSurface(
    const GeometryContext &gctx, const PlaneSurface<surface_bounds_t> &other,
    const Transform3D &transf)
    : GeometryObject(), Surface(gctx, other, transf), m_bounds(other.m_bounds) {
}
template <typename surface_bounds_t>
inline PlaneSurface<surface_bounds_t>::PlaneSurface(const Vector3D &center,
                                                    const Vector3D &normal)
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
  m_transform = transform;
}

template <typename surface_bounds_t>
inline PlaneSurface<surface_bounds_t>::PlaneSurface(
    const Vector3D &center, const Vector3D &normal,
    const HomogeneousSurfaceMaterial &material)
    : Surface(), m_bounds(nullptr) {
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
  m_transform = transform;
  // Set the surface material
  m_surfaceMaterial = material;
}

template <typename surface_bounds_t>
template <typename T,
          std::enable_if_t<not std::is_same<T, InfiniteBounds>::value, int>>
inline PlaneSurface<surface_bounds_t>::PlaneSurface(
    const Transform3D &htrans, const surface_bounds_t *pbounds)
    : Surface(std::move(htrans)), m_bounds(std::move(pbounds)) {}

template <typename surface_bounds_t>
inline PlaneSurface<surface_bounds_t> &Acts::PlaneSurface<surface_bounds_t>::
operator=(const PlaneSurface<surface_bounds_t> &other) {
  if (this != &other) {
    Surface::operator=(other);
    m_bounds = other.m_bounds;
  }
  return *this;
}

template <typename surface_bounds_t>
inline Surface::SurfaceType PlaneSurface<surface_bounds_t>::type() const {
  return Surface::Plane;
}

template <typename surface_bounds_t>
const surface_bounds_t *PlaneSurface<surface_bounds_t>::bounds() const {
  return m_bounds;
}

// inline void PlaneSurface::localToGlobal(const GeometryContext &gctx,
//                                       const Vector2D &lposition,
//                                       const Vector3D & /*gmom*/,
//                                       Vector3D &position) const {
//  Vector3D loc3Dframe(lposition[eLOC_X], lposition[eLOC_Y], 0.);
//  /// the chance that there is no transform is almost 0, let's apply it
//  position = transform(gctx) * loc3Dframe;
//}
//
// inline bool PlaneSurface::globalToLocal(const GeometryContext &gctx,
//                                       const Vector3D &position,
//                                       const Vector3D & /*gmom*/,
//                                       Acts::Vector2D &lposition) const {
//  /// the chance that there is no transform is almost 0, let's apply it
//  Vector3D loc3Dframe = (transform(gctx).inverse()) * position;
//  lposition = Vector2D(loc3Dframe.x(), loc3Dframe.y());
//  return ((loc3Dframe.z() * loc3Dframe.z() >
//           s_onSurfaceTolerance * s_onSurfaceTolerance)
//              ? false
//              : true);
//}
//
// inline const Vector3D PlaneSurface::normal(const GeometryContext &gctx,
//                                           const Vector2D & /*lpos*/) const {
//  // fast access via tranform matrix (and not rotation())
//  const auto &tMatrix = transform(gctx).matrix();
//  return Vector3D(tMatrix(0, 2), tMatrix(1, 2), tMatrix(2, 2));
//}

template <typename surface_bounds_t>
inline const Vector3D
PlaneSurface<surface_bounds_t>::binningPosition(const GeometryContext &gctx,
                                                BinningValue /*bValue*/) const {
  return center(gctx);
}

/*
inline double PlaneSurface::pathCorrection(const GeometryContext &gctx,
                                           const Vector3D &position,
                                           const Vector3D &direction) const {
  // We can ignore the global position here
  return 1. / std::abs(Surface::normal(gctx, position).dot(direction));
}
*/

template <typename surface_bounds_t>
inline SurfaceIntersection PlaneSurface<surface_bounds_t>::intersect(
    const GeometryContext &gctx, const Vector3D &position,
    const Vector3D &direction, const BoundaryCheck &bcheck) const {
  // Get the contextual transform
  const auto &gctxTransform = transform(gctx);
  // Use the intersection helper for planar surfaces
  auto intersection =
      PlanarHelper::intersectionEstimate(gctxTransform, position, direction);
  // Evaluate boundary check if requested (and reachable)
  if (intersection.status != Intersection::Status::unreachable and bcheck) {
    // Built-in local to global for speed reasons
    const auto &tMatrix = gctxTransform.matrix();
    // Create the reference vector in local
    const Vector3D vecLocal(intersection.position - tMatrix.block<3, 1>(0, 3));
    if (not insideBounds<PlaneSurface<surface_bounds_t>>(
            tMatrix.block<3, 2>(0, 0).transpose() * vecLocal, bcheck)) {
      intersection.status = Intersection::Status::missed;
    }
  }
  return {intersection, this};
}
