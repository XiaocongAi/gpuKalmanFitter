// This file is part of the Acts project.
//
// Copyright (C) 2018-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

inline LineSurface::LineSurface(const LineSurface &other)
    : GeometryObject(), Surface(other) {}

inline LineSurface::LineSurface(const GeometryContext &gctx,
                                const LineSurface &other,
                                const Transform3D &shift)
    : GeometryObject(), Surface(gctx, other, shift) {}

inline LineSurface::LineSurface(const Transform3D &transform)
    : GeometryObject(), Surface(transform) {}

inline LineSurface::LineSurface(const Vector3D &gp) {
  Surface::m_transform = Transform3D(Translation3D(gp.x(), gp.y(), gp.z()));
}

inline LineSurface &LineSurface::operator=(const LineSurface &other) {
  if (this != &other) {
    Surface::operator=(other);
  }
  return *this;
}

inline Vector3D LineSurface::normal(const GeometryContext &gctx,
                                    const Vector2D & /*lpos*/) const {
  const auto &tMatrix = transform(gctx).matrix();
  return Vector3D(tMatrix(0, 2), tMatrix(1, 2), tMatrix(2, 2));
}

inline const Vector3D
LineSurface::binningPosition(const GeometryContext &gctx,
                             BinningValue /*bValue*/) const {
  return center(gctx);
}

inline Surface::SurfaceType LineSurface::type() const { return Surface::Line; }

inline const InfiniteBounds *LineSurface::bounds() const { return nullptr; }

inline RotationMatrix3D const
LineSurface::referenceFrame(const GeometryContext &gctx,
                            const Vector3D & /*unused*/,
                            const Vector3D &momentum) const {
  RotationMatrix3D mFrame;
  const auto &tMatrix = transform(gctx).matrix();
  Vector3D measY(tMatrix(0, 2), tMatrix(1, 2), tMatrix(2, 2));
  Vector3D measX(measY.cross(momentum).normalized());
  Vector3D measDepth(measX.cross(measY));
  // assign the columnes
  mFrame.col(0) = measX;
  mFrame.col(1) = measY;
  mFrame.col(2) = measDepth;
  // return the rotation matrix
  return mFrame;
}

inline void LineSurface::initJacobianToGlobal(const GeometryContext &gctx,
                                              BoundToFreeMatrix &jacobian,
                                              const Vector3D &position,
                                              const Vector3D &direction,
                                              const BoundVector &pars) const {
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
  // the local error components - given by the reference frame
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

  // the projection of direction onto ref frame normal
  double ipdn = 1. / direction.dot(rframe.col(2));
  // build the cross product of d(D)/d(ePHI) components with y axis
  auto dDPhiY = rframe.block<3, 1>(0, 1).cross(jacobian.block<3, 1>(4, ePHI));
  // and the same for the d(D)/d(eTheta) components
  auto dDThetaY =
      rframe.block<3, 1>(0, 1).cross(jacobian.block<3, 1>(4, eTHETA));
  // and correct for the x axis components
  dDPhiY -= rframe.block<3, 1>(0, 0) * (rframe.block<3, 1>(0, 0).dot(dDPhiY));
  dDThetaY -=
      rframe.block<3, 1>(0, 0) * (rframe.block<3, 1>(0, 0).dot(dDThetaY));
  // set the jacobian components for global d/ phi/Theta
  jacobian.block<3, 1>(0, ePHI) = dDPhiY * pars[eLOC_0] * ipdn;
  jacobian.block<3, 1>(0, eTHETA) = dDThetaY * pars[eLOC_0] * ipdn;
}

inline const RotationMatrix3D LineSurface::initJacobianToLocal(
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

inline const BoundRowVector LineSurface::derivativeFactors(
    const GeometryContext &gctx, const Vector3D &position,
    const Vector3D &direction, const RotationMatrix3D &rft,
    const BoundToFreeMatrix &jacobian) const {
  // the vector between position and center
  ActsRowVectorD<3> pc = (position - center(gctx)).transpose();
  // the longitudinal component vector (alogn local z)
  ActsRowVectorD<3> locz = rft.block<1, 3>(1, 0);
  // build the norm vector comonent by subtracting the longitudinal one
  double long_c = locz * direction;
  ActsRowVectorD<3> norm_vec = direction.transpose() - long_c * locz;
  // calculate the s factors for the dependency on X
  const BoundRowVector s_vec =
      norm_vec * jacobian.topLeftCorner<3, eBoundParametersSize>();
  // calculate the d factors for the dependency on Tx
  const BoundRowVector d_vec =
      locz * jacobian.block<3, eBoundParametersSize>(4, 0);
  // normalisation of normal & longitudinal components
  double norm = 1. / (1. - long_c * long_c);
  // create a matrix representation
  ActsMatrixD<3, eBoundParametersSize> long_mat =
      ActsMatrixD<3, eBoundParametersSize>::Zero();
  long_mat.colwise() += locz.transpose();
  // build the combined normal & longitudinal components
  return (norm *
          (s_vec - pc * (long_mat * d_vec.asDiagonal() -
                         jacobian.block<3, eBoundParametersSize>(4, 0))));
}

inline void LineSurface::localToGlobal(const GeometryContext &gctx,
                                       const Vector2D &lposition,
                                       const Vector3D &momentum,
                                       Vector3D &position) const {
  const auto &sTransform = transform(gctx);
  const auto &tMatrix = sTransform.matrix();
  Vector3D lineDirection(tMatrix(0, 2), tMatrix(1, 2), tMatrix(2, 2));

  // get the vector perpendicular to the momentum and the straw axis
  Vector3D radiusAxisGlobal(lineDirection.cross(momentum));
  Vector3D locZinGlobal = sTransform * Vector3D(0., 0., lposition[eLOC_Z]);
  // add eLOC_R * radiusAxis
  position = Vector3D(locZinGlobal +
                      lposition[eLOC_R] * radiusAxisGlobal.normalized());
}

inline bool LineSurface::globalToLocal(const GeometryContext &gctx,
                                       const Vector3D &position,
                                       const Vector3D &momentum,
                                       Vector2D &lposition) const {
  using VectorHelpers::perp;

  const auto &sTransform = transform(gctx);
  const auto &tMatrix = sTransform.matrix();
  Vector3D lineDirection(tMatrix(0, 2), tMatrix(1, 2), tMatrix(2, 2));
  // Bring the global position into the local frame
  Vector3D loc3Dframe = sTransform.inverse() * position;
  // construct localPosition with sign*perp(candidate) and z.()
  lposition = Vector2D(perp(loc3Dframe), loc3Dframe.z());
  Vector3D sCenter(tMatrix(0, 3), tMatrix(1, 3), tMatrix(2, 3));
  Vector3D decVec(position - sCenter);
  // assign the right sign
  double sign = ((lineDirection.cross(momentum)).dot(decVec) < 0.) ? -1. : 1.;
  lposition[eLOC_R] *= sign;
  return true;
}

inline double LineSurface::pathCorrection(const GeometryContext & /*unused*/,
                                          const Vector3D & /*pos*/,
                                          const Vector3D & /*mom*/) const {
  return 1.;
}

inline SurfaceIntersection
LineSurface::intersect(const GeometryContext &gctx, const Vector3D &position,
                       const Vector3D &direction,
                       const BoundaryCheck & /*bcheck*/) const {
  // following nominclature found in header file and doxygen documentation
  // line one is the straight track
  const Vector3D &ma = position;
  const Vector3D &ea = direction;
  // line two is the line surface
  const auto &tMatrix = transform(gctx).matrix();
  Vector3D mb = tMatrix.block<3, 1>(0, 3).transpose();
  Vector3D eb = tMatrix.block<3, 1>(0, 2).transpose();
  // now go ahead and solve for the closest approach
  Vector3D mab(mb - ma);
  double eaTeb = ea.dot(eb);
  double denom = 1 - eaTeb * eaTeb;
  // validity parameter
  Intersection::Status status = Intersection::Status::unreachable;
  if (denom * denom > s_onSurfaceTolerance * s_onSurfaceTolerance) {
    double u = (mab.dot(ea) - mab.dot(eb) * eaTeb) / denom;
    // Check if we are on the surface already
    status = (u * u < s_onSurfaceTolerance * s_onSurfaceTolerance)
                 ? Intersection::Status::onSurface
                 : Intersection::Status::reachable;
    Vector3D result = (ma + u * ea);
    // Evaluate the boundary check if requested
    // m_bounds == nullptr prevents unecessary calulations for PerigeeSurface
    return {Intersection(result, u, status), this};
  }
  // return a false intersection
  return {Intersection(position, std::numeric_limits<double>::max(), status),
          this};
}

// inline std::string LineSurface::name() const { return "Acts::LineSurface"; }
