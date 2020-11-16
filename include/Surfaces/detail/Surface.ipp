// This file is part of the Acts project.
//
// Copyright (C) 2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

inline Surface::Surface(const Transform3D &tform)
    : GeometryObject(), m_transform(std::move(tform)) {}

inline Surface::Surface(const Surface &other)
    : GeometryObject(other), m_transform(other.m_transform),
      m_surfaceMaterial(other.m_surfaceMaterial) {}

inline Surface::Surface(const GeometryContext &gctx, const Surface &other,
                        const Transform3D &shift)
    : GeometryObject(), m_transform(Transform3D(shift * other.transform(gctx))),
      m_surfaceMaterial(other.m_surfaceMaterial) {}

template <typename Derived> inline Surface::SurfaceType Surface::type() const {
  return static_cast<const Derived *>(this)->type();
}

template <typename Derived>
inline bool Surface::isOnSurface(const GeometryContext &gctx,
                                 const Vector3D &position,
                                 const Vector3D &momentum,
                                 const BoundaryCheck &bcheck) const {
  // create the local position
  Vector2D lposition{0., 0.};
  // global to local transformation
  bool gtlSuccess = globalToLocal(gctx, position, momentum, lposition);
  if (gtlSuccess) {
    // No bounds
    if (bounds<Derived>() == nullptr) {
      return true;
    }
    return bcheck ? bounds<Derived>()->inside(lposition, bcheck) : true;
  }
  // did not succeed
  return false;
}

template <typename Derived>
inline const typename Derived::SurfaceBoundsType *Surface::bounds() const {
  return static_cast<const Derived *>(this)->bounds();
}

inline Surface &Surface::operator=(const Surface &other) {
  if (&other != this) {
    //@Todo: active this
    // GeometryObject::operator=(other);
    // detector element, identifier & layer association are unique
    m_surfaceMaterial = other.m_surfaceMaterial;
    m_transform = other.m_transform;
  }
  return *this;
}

inline bool Surface::operator==(const Surface &other) const {
  // (a) fast exit for pointer comparison
  if (&other == this) {
    return true;
  }
  //  // (b) fast exit for type
  //    if (other.type<Derived>() != type<Derived>()) {
  //      return false;
  //    }
  //  // (c) fast exit for bounds
  //    if (*other.bounds<Derived>() != *bounds<Derived>()) {
  //      return false;
  //    }
  // (e) compare transform values
  if (!m_transform.isApprox(other.m_transform, 1e-9)) {
    return false;
  }
  if (!(m_surfaceMaterial == other.m_surfaceMaterial)) {
    return false;
  }

  // we should be good
  return true;
}

inline bool Surface::operator!=(const Surface &sf) const {
  return !(operator==(sf));
}

inline const Vector3D Surface::center(const GeometryContext &gctx) const {
  // fast access via tranform matrix (and not translation())
  auto tMatrix = m_transform.matrix();
  return Vector3D(tMatrix(0, 3), tMatrix(1, 3), tMatrix(2, 3));
}

inline const Acts::Vector3D Surface::normal(const GeometryContext &gctx,
                                            const Vector3D & /*unused*/) const {
  return normal(gctx, Vector2D(0, 0));
}

inline const Transform3D &
Surface::transform(const GeometryContext &gctx) const {
  return m_transform;
}

template <typename Derived>
inline bool Surface::insideBounds(const Vector2D &lposition,
                                  const BoundaryCheck &bcheck) const {
  if (bounds<Derived>() == nullptr) {
    return true;
  }
  return bounds<Derived>()->inside(lposition, bcheck);
}

template <typename Derived>
inline const RotationMatrix3D
Surface::referenceFrame(const GeometryContext &gctx, const Vector3D &position,
                        const Vector3D &direction) const {
  return static_cast<const Derived *>(this)->referenceFrame(gctx, position,
                                                            direction);
}

template <typename Derived>
inline ACTS_DEVICE_FUNC void Surface::initJacobianToGlobal(
    const GeometryContext &gctx, BoundToFreeMatrix &jacobian,
    const Vector3D &position, const Vector3D &direction,
    const BoundVector &pars) const {
  static_cast<const Derived *>(this)->initJacobianToGlobal(
      gctx, jacobian, position, direction, pars);
}

template <typename Derived>
inline const RotationMatrix3D Surface::initJacobianToLocal(
    const GeometryContext &gctx, FreeToBoundMatrix &jacobian,
    const Vector3D &position, const Vector3D &direction) const {
  return static_cast<const Derived *>(this)->initJacobianToLocal(
      gctx, jacobian, position, direction);
}

template <typename Derived>
inline const BoundRowVector
Surface::derivativeFactors(const GeometryContext &gctx,
                           const Vector3D &position, const Vector3D &direction,
                           const RotationMatrix3D &rft,
                           const BoundToFreeMatrix &jacobian) const {
  return static_cast<const Derived *>(this)->derivativeFactors(
      gctx, position, direction, rft, jacobian);
}

inline void Surface::localToGlobal(const GeometryContext &gctx,
                                   const Vector2D &lposition,
                                   const Vector3D & /*gmom*/,
                                   Vector3D &position) const {
  Vector3D loc3Dframe(lposition[eLOC_X], lposition[eLOC_Y], 0.);
  /// the chance that there is no transform is almost 0, let's apply it
  position = transform(gctx) * loc3Dframe;
}

inline bool Surface::globalToLocal(const GeometryContext &gctx,
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

inline const Vector3D Surface::normal(const GeometryContext &gctx,
                                      const Vector2D & /*lpos*/) const {
  // fast access via tranform matrix (and not rotation())
  const auto &tMatrix = transform(gctx).matrix();
  return Vector3D(tMatrix(0, 2), tMatrix(1, 2), tMatrix(2, 2));
}

inline double Surface::pathCorrection(const GeometryContext &gctx,
                                      const Vector3D &position,
                                      const Vector3D &direction) const {
  // We can ignore the global position here
  return 1. / std::abs(normal(gctx, position).dot(direction));
}

template <typename Derived>
inline SurfaceIntersection
Surface::intersect(const GeometryContext &gctx, const Vector3D &position,
                   const Vector3D &direction,
                   const BoundaryCheck &bcheck) const {
  return static_cast<const Derived *>(this)->intersect(gctx, position,
                                                       direction, bcheck);
}

inline const Acts::HomogeneousSurfaceMaterial &
Acts::Surface::surfaceMaterial() const {
  return m_surfaceMaterial;
}
