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
  bool gtlSuccess = globalToLocal<Derived>(gctx, position, momentum, lposition);
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
    // detector element, identifier & layer association are unique
    GeometryObject::operator=(other);
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

template <typename Derived>
inline const Acts::Vector3D Surface::normal(const GeometryContext &gctx,
                                            const Vector3D & /*unused*/) const {
  return normal<Derived>(gctx, Vector2D(0, 0));
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

template <typename Derived>
inline void
Surface::localToGlobal(const GeometryContext &gctx, const Vector2D &lposition,
                       const Vector3D &momentum, Vector3D &position) const {
  static_cast<const Derived *>(this)->localToGlobal(gctx, lposition, momentum,
                                                    position);
}

template <typename Derived>
inline bool Surface::globalToLocal(const GeometryContext &gctx,
                                   const Vector3D &position,
                                   const Vector3D &momentum,
                                   Acts::Vector2D &lposition) const {
  return static_cast<const Derived *>(this)->globalToLocal(gctx, position,
                                                           momentum, lposition);
}

template <typename Derived>
inline const Vector3D Surface::normal(const GeometryContext &gctx,
                                      const Vector2D &lpos) const {
  return static_cast<const Derived *>(this)->normal(gctx, lpos);
}

template <typename Derived>
inline ActsScalar Surface::pathCorrection(const GeometryContext &gctx,
                                          const Vector3D &position,
                                          const Vector3D &direction) const {
  return static_cast<const Derived *>(this)->pathCorrection(gctx, position,
                                                            direction);
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
