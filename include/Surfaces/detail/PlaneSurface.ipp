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
    : Surface(other), m_bounds(other.m_bounds) {}

template <typename surface_bounds_t>
inline PlaneSurface<surface_bounds_t>::PlaneSurface(
    const GeometryContext &gctx, const PlaneSurface<surface_bounds_t> &other,
    const Transform3D &transf)
    : Surface(gctx, other, transf), m_bounds(other.m_bounds) {}

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
inline PlaneSurface<surface_bounds_t> &
Acts::PlaneSurface<surface_bounds_t>::operator=(
    const PlaneSurface<surface_bounds_t> &other) {
  if (this != &other) {
    Surface::operator=(other);
    m_bounds = other.m_bounds;
  }
  return *this;
}

template <typename surface_bounds_t>
inline const Vector3D
PlaneSurface<surface_bounds_t>::normal(const GeometryContext &gctx,
                                       const Vector2D & /*lpos*/) const {
  // fast access via tranform matrix (and not rotation())
  const auto &tMatrix = transform(gctx).matrix();
  return Vector3D(tMatrix(0, 2), tMatrix(1, 2), tMatrix(2, 2));
}

template <typename surface_bounds_t>
inline const Vector3D
PlaneSurface<surface_bounds_t>::binningPosition(const GeometryContext &gctx,
                                                BinningValue /*bValue*/) const {
  return center(gctx);
}

template <typename surface_bounds_t>
inline Surface::SurfaceType PlaneSurface<surface_bounds_t>::type() const {
  return Surface::Plane;
}

template <typename surface_bounds_t>
const surface_bounds_t *PlaneSurface<surface_bounds_t>::bounds() const {
  return m_bounds;
}

template <typename surface_bounds_t>
inline const RotationMatrix3D PlaneSurface<surface_bounds_t>::referenceFrame(
    const GeometryContext &gctx, const Vector3D & /*unused*/,
    const Vector3D & /*unused*/) const {
  return transform(gctx).matrix().block<3, 3>(0, 0);
}

template <typename surface_bounds_t>
inline void PlaneSurface<surface_bounds_t>::initJacobianToGlobal(
    const GeometryContext &gctx, BoundToFreeMatrix &jacobian,
    const Vector3D &position, const Vector3D &direction,
    const BoundVector & /*pars*/) const {
  // The trigonometry required to convert the direction to spherical
  // coordinates and then compute the sines and cosines again can be
  // surprisingly expensive from a performance point of view.
  //
  // Here, we can avoid it because the direction is by definition a unit
  // vector, with the following coordinate conversions...
  const ActsScalar x = direction(0); // == cos(phi) * sin(theta)
  const ActsScalar y = direction(1); // == sin(phi) * sin(theta)
  const ActsScalar z = direction(2); // == cos(theta)

  // ...which we can invert to directly get the sines and cosines:
  const ActsScalar cos_theta = z;
  const ActsScalar sin_theta = sqrt(x * x + y * y);
  const ActsScalar inv_sin_theta = 1. / sin_theta;
  const ActsScalar cos_phi = x * inv_sin_theta;
  const ActsScalar sin_phi = y * inv_sin_theta;
  // retrieve the reference frame
  const auto rframe = referenceFrame(gctx, position, direction);
  // the local error components - given by reference frame
  jacobian.topLeftCorner<3, 2>() = rframe.template topLeftCorner<3, 2>();
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

template <typename surface_bounds_t>
inline const RotationMatrix3D
PlaneSurface<surface_bounds_t>::initJacobianToLocal(
    const GeometryContext &gctx, FreeToBoundMatrix &jacobian,
    const Vector3D &position, const Vector3D &direction) const {
  // Optimized trigonometry on the propagation direction
  const ActsScalar x = direction(0); // == cos(phi) * sin(theta)
  const ActsScalar y = direction(1); // == sin(phi) * sin(theta)
  const ActsScalar z = direction(2); // == cos(theta)
  // can be turned into cosine/sine
  const ActsScalar cosTheta = z;
  const ActsScalar sinTheta = sqrt(x * x + y * y);
  const ActsScalar invSinTheta = 1. / sinTheta;
  const ActsScalar cosPhi = x * invSinTheta;
  const ActsScalar sinPhi = y * invSinTheta;
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

template <typename surface_bounds_t>
inline const BoundRowVector PlaneSurface<surface_bounds_t>::derivativeFactors(
    const GeometryContext & /*unused*/, const Vector3D & /*unused*/,
    const Vector3D &direction, const RotationMatrix3D &rft,
    const BoundToFreeMatrix &jacobian) const {
  // Create the normal and scale it with the projection onto the direction
  ActsRowVectorD<3> norm_vec = rft.template block<1, 3>(2, 0);
  const ActsScalar dotProduct = norm_vec[0] * direction[0] +
                                norm_vec[1] * direction[1] +
                                norm_vec[2] * direction[2];
  norm_vec /= dotProduct;
  // calculate the s factors
  return (norm_vec * jacobian.topLeftCorner<3, eBoundParametersSize>());
}

template <typename surface_bounds_t>
inline void PlaneSurface<surface_bounds_t>::localToGlobal(
    const GeometryContext &gctx, const Vector2D &lposition,
    const Vector3D & /*gmom*/, Vector3D &position) const {
  Vector3D loc3Dframe(lposition[eLOC_X], lposition[eLOC_Y], 0.);
  /// the chance that there is no transform is almost 0, let's apply it
  position = transform(gctx) * loc3Dframe;
}

template <typename surface_bounds_t>
inline bool PlaneSurface<surface_bounds_t>::globalToLocal(
    const GeometryContext &gctx, const Vector3D &position,
    const Vector3D & /*gmom*/, Acts::Vector2D &lposition) const {
  /// the chance that there is no transform is almost 0, let's apply it
  Vector3D loc3Dframe = (transform(gctx).inverse()) * position;
  lposition = Vector2D(loc3Dframe.x(), loc3Dframe.y());
  return ((loc3Dframe.z() * loc3Dframe.z() >
           s_onSurfaceTolerance * s_onSurfaceTolerance)
              ? false
              : true);
}

template <typename surface_bounds_t>
inline ActsScalar PlaneSurface<surface_bounds_t>::pathCorrection(
    const GeometryContext &gctx, const Vector3D &position,
    const Vector3D &direction) const {
  // We can ignore the global position here
  return 1. / std::abs(Surface::normal<PlaneSurface<surface_bounds_t>>(gctx,
                                                                       position)
                           .dot(direction));
}

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
