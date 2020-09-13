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
    : GeometryObject(other), m_transform(other.m_transform) {}

inline Surface::Surface(const GeometryContext &gctx, const Surface &other,
                        const Transform3D &shift)
    : GeometryObject(),
      m_transform(Transform3D(shift * other.transform(gctx))) {}

inline Surface::Surface(const Vector3D &center, const Vector3D &normal) {
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

inline const RotationMatrix3D
Surface::referenceFrame(const GeometryContext &gctx,
                        const Vector3D & /*unused*/,
                        const Vector3D & /*unused*/) const {
  return transform(gctx).matrix().block<3, 3>(0, 0);
}

inline ACTS_DEVICE_FUNC void Surface::initJacobianToGlobal(
    const GeometryContext &gctx, BoundToFreeMatrix &jacobian,
    const Vector3D &position, const Vector3D &direction,
    const BoundVector & /*pars*/) const {
  // The trigonometry required to convert the direction to spherical
  // coordinates and then compute the sines and cosines again can be
  // surprisingly expensive from a performance point of view.
  //
  // Here, we can avoid it because the direction is by definition a unit
  // vector, with the following coordinate conversions...
  const double x = direction(0); // == cos(phi) * sin(theta)
  const double y = direction(1); // == sin(phi) * sin(theta)
  const double z = direction(2); // == cos(theta)

  // ...which we can invert to directly get the sines and cosines:
  const double cos_theta = z;
  const double sin_theta = sqrt(x * x + y * y);
  const double inv_sin_theta = 1. / sin_theta;
  const double cos_phi = x * inv_sin_theta;
  const double sin_phi = y * inv_sin_theta;
  // retrieve the reference frame
  const auto rframe = referenceFrame(gctx, position, direction);
  // the local error components - given by reference frame
  jacobian.topLeftCorner<3, 2>() = rframe.topLeftCorner<3, 2>();
  // the time component
  jacobian(3, eT) = 1;
  // the momentum components
  jacobian(4, ePHI) = (-sin_theta) * sin_phi;
  jacobian(4, eTHETA) = cos_theta * cos_phi;
  jacobian(5, ePHI) = sin_theta * cos_phi;
  jacobian(5, eTHETA) = cos_theta * sin_phi;
  jacobian(6, eTHETA) = (-sin_theta);
  jacobian(7, eQOP) = 1;
}

inline const RotationMatrix3D Surface::initJacobianToLocal(
    const GeometryContext &gctx, FreeToBoundMatrix &jacobian,
    const Vector3D &position, const Vector3D &direction) const {
  // Optimized trigonometry on the propagation direction
  const double x = direction(0); // == cos(phi) * sin(theta)
  const double y = direction(1); // == sin(phi) * sin(theta)
  const double z = direction(2); // == cos(theta)
  // can be turned into cosine/sine
  const double cosTheta = z;
  const double sinTheta = sqrt(x * x + y * y);
  const double invSinTheta = 1. / sinTheta;
  const double cosPhi = x * invSinTheta;
  const double sinPhi = y * invSinTheta;
  // The measurement frame of the surface
  RotationMatrix3D rframeT =
      referenceFrame(gctx, position, direction).transpose();
  // given by the refernece frame
  jacobian.block<2, 3>(0, 0) = rframeT.block<2, 3>(0, 0);
  // Time component
  jacobian(eT, 3) = 1;
  // Directional and momentum elements for reference frame surface
  jacobian(ePHI, 4) = -sinPhi * invSinTheta;
  jacobian(ePHI, 5) = cosPhi * invSinTheta;
  jacobian(eTHETA, 4) = cosPhi * cosTheta;
  jacobian(eTHETA, 5) = sinPhi * cosTheta;
  jacobian(eTHETA, 6) = -sinTheta;
  jacobian(eQOP, 7) = 1;
  // return the frame where this happened
  return rframeT;
}

inline const BoundRowVector Surface::derivativeFactors(
    const GeometryContext & /*unused*/, const Vector3D & /*unused*/,
    const Vector3D &direction, const RotationMatrix3D &rft,
    const BoundToFreeMatrix &jacobian) const {
  // Create the normal and scale it with the projection onto the direction
  ActsRowVectorD<3> norm_vec = rft.template block<1, 3>(2, 0);
  const double dotProduct = norm_vec[0] * direction[0] +
                            norm_vec[1] * direction[1] +
                            norm_vec[2] * direction[2];
  norm_vec /= dotProduct;
  // calculate the s factors
  return (norm_vec * jacobian.topLeftCorner<3, eBoundParametersSize>());
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
inline Intersection Surface::intersectionEstimate(
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
    if (not insideBounds<Derived>(
            tMatrix.block<3, 2>(0, 0).transpose() * vecLocal, bcheck)) {
      intersection.status = Intersection::Status::missed;
    }
  }
  return intersection;
}
