// This file is part of the Acts project.
//
// Copyright (C) 2019-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EventData/TrackParameters.hpp"
#include "Geometry/GeometryContext.hpp"
#include "Surfaces/Surface.hpp"
#include "Utilities/Definitions.hpp"

#include <cmath>
#include <functional>

namespace Acts {

// struct BoundState {
//  BoundParameters boundParams;
//  BoundMatrix jacobian;
//  double path;
//};
//

struct CurvilinearState {
  CurvilinearParameters curvParams;
  BoundMatrix jacobian;
  double path;
};

namespace {
/// Some type defs
using Jacobian = BoundMatrix;
using Covariance = BoundSymMatrix;

/// @brief Evaluate the projection Jacobian from free to curvilinear parameters
///
/// @param [in] direction Normalised direction vector
///
/// @return Projection Jacobian
ACTS_DEVICE_FUNC FreeToBoundMatrix
freeToCurvilinearJacobian(const Vector3D &direction) {
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
  // prepare the jacobian to curvilinear
  FreeToBoundMatrix jacToCurv = FreeToBoundMatrix::Zero();
  if (std::abs(cosTheta) < s_curvilinearProjTolerance) {
    // We normally operate in curvilinear coordinates defined as follows
    jacToCurv(0, 0) = -sinPhi;
    jacToCurv(0, 1) = cosPhi;
    jacToCurv(1, 0) = -cosPhi * cosTheta;
    jacToCurv(1, 1) = -sinPhi * cosTheta;
    jacToCurv(1, 2) = sinTheta;
  } else {
    // Under grazing incidence to z, the above coordinate system definition
    // becomes numerically unstable, and we need to switch to another one
    const double c = sqrt(y * y + z * z);
    const double invC = 1. / c;
    jacToCurv(0, 1) = -z * invC;
    jacToCurv(0, 2) = y * invC;
    jacToCurv(1, 0) = c;
    jacToCurv(1, 1) = -x * y * invC;
    jacToCurv(1, 2) = -x * z * invC;
  }
  // Time parameter
  jacToCurv(5, 3) = 1.;
  // Directional and momentum parameters for curvilinear
  jacToCurv(2, 4) = -sinPhi * invSinTheta;
  jacToCurv(2, 5) = cosPhi * invSinTheta;
  jacToCurv(3, 4) = cosPhi * cosTheta;
  jacToCurv(3, 5) = sinPhi * cosTheta;
  jacToCurv(3, 6) = -sinTheta;
  jacToCurv(4, 7) = 1.;

  return jacToCurv;
}

/// @brief This function treats the modifications of the jacobian related to the
/// projection onto a surface. Since a variation of the start parameters within
/// a given uncertainty would lead to a variation of the end parameters, these
/// need to be propagated onto the target surface. This an approximated approach
/// to treat the (assumed) small change.
///
/// @param [in] geoContext The geometry Context
/// @param [in] parameters Free, nominal parametrisation
/// @param [in] jacobianLocalToGlobal The projection jacobian from local start
/// to global final parameters
/// @param [in, out] derivatives Path length derivatives of the free, nominal
/// parameters
/// @param [in] surface The surface onto which the projection should be
/// performed
///
/// @return The projection jacobian from global end parameters to its local
/// equivalentconst
template <typename surface_derived_t>
ACTS_DEVICE_FUNC FreeToBoundMatrix surfaceDerivative(
    const GeometryContext &geoContext, const FreeVector &parameters,
    BoundToFreeMatrix &jacobianLocalToGlobal, const FreeVector &derivatives,
    const Surface &surface) {
  // Initialize the transport final frame jacobian
  FreeToBoundMatrix jacToLocal = FreeToBoundMatrix::Zero();
  // Initalize the jacobian to local, returns the transposed ref frame
  auto rframeT = surface.initJacobianToLocal<surface_derived_t>(
      geoContext, jacToLocal, parameters.segment<3>(eFreePos0),
      parameters.segment<3>(eFreeDir0));
  // Calculate the form factors for the derivatives
  const BoundRowVector sVec = surface.derivativeFactors<surface_derived_t>(
      geoContext, parameters.segment<3>(eFreePos0),
      parameters.segment<3>(eFreeDir0), rframeT, jacobianLocalToGlobal);
  // 8*6 = 8*1 * 1*6
  jacobianLocalToGlobal -= derivatives * sVec;
  // Return the jacobian to local
  return jacToLocal;
}

/// @brief This function treats the modifications of the jacobian related to the
/// projection onto a curvilinear surface. Since a variation of the start
/// parameters within a given uncertainty would lead to a variation of the end
/// parameters, these need to be propagated onto the target surface. This an
/// approximated approach to treat the (assumed) small change.
///
/// @param [in] direction Normalised direction vector
/// @param [in] jacobianLocalToGlobal The projection jacobian from local start
/// to global final parameters
/// @param [in, out] derivatives Path length derivatives of the free, nominal
/// parameters
/// @note The parameter @p surface is only required if projected to bound
/// parameters. In the case of curvilinear parameters the geometry and the
/// position is known and the calculation can be simplified
///
/// @return The projection jacobian from global end parameters to its local
/// equivalent
ACTS_DEVICE_FUNC const FreeToBoundMatrix surfaceDerivative(
    const Vector3D &direction, BoundToFreeMatrix &jacobianLocalToGlobal,
    const FreeVector &derivatives) {
  // Transport the covariance
  const ActsRowVectorD<3> normVec(direction);
  const BoundRowVector sfactors =
      normVec *
      jacobianLocalToGlobal.template topLeftCorner<3, eBoundParametersSize>();
  jacobianLocalToGlobal -= derivatives * sfactors;
  // Since the jacobian to local needs to calculated for the bound parameters
  // here, it is convenient to do the same here
  return freeToCurvilinearJacobian(direction);
}

/// @brief This function reinitialises the state members required for the
/// covariance transport
///
/// @param [in] geoContext The geometry context
/// @param [in, out] jacobian Full jacobian since the last reset
/// @param [in, out] derivatives Path length derivatives of the free, nominal
/// parameters
/// @param [in, out] jacobianLocalToGlobal Projection jacobian of the last bound
/// parametrisation to free parameters
/// @param [in] parameters Free, nominal parametrisation
/// @param [in] surface The surface the represents the local parametrisation
template <typename surface_derived_t>
ACTS_DEVICE_FUNC void
reinitializeJacobians(const GeometryContext &geoContext,
                      FreeMatrix &transportJacobian, FreeVector &derivatives,
                      BoundToFreeMatrix &jacobianLocalToGlobal,
                      const FreeVector &parameters, const Surface &surface) {
  using VectorHelpers::phi;
  using VectorHelpers::theta;

  // Reset the jacobians
  transportJacobian = FreeMatrix::Identity();
  derivatives = FreeVector::Zero();
  jacobianLocalToGlobal = BoundToFreeMatrix::Zero();

  // Reset the jacobian from local to global
  Vector2D loc{0., 0.};
  const Vector3D position = parameters.segment<3>(eFreePos0);
  const Vector3D direction = parameters.segment<3>(eFreeDir0);
  surface.globalToLocal<surface_derived_t>(geoContext, position, direction,
                                           loc);
  BoundVector pars;
  //  pars << loc[eLOC_0], loc[eLOC_1], phi(direction), theta(direction),
  //      parameters[eFreeQOverP], parameters[eFreeTime];
  pars[0] = loc[eLOC_0];
  pars[1] = loc[eLOC_1];
  pars[2] = phi(direction);
  pars[3] = theta(direction);
  pars[4] = parameters[eFreeQOverP];
  pars[5] = parameters[eFreeTime];

  surface.initJacobianToGlobal<surface_derived_t>(
      geoContext, jacobianLocalToGlobal, position, direction, pars);
}

/// @brief This function reinitialises the state members required for the
/// covariance transport
///
/// @param [in, out] jacobian Full jacobian since the last reset
/// @param [in, out] derivatives Path length derivatives of the free, nominal
/// parameters
/// @param [in, out] jacobianLocalToGlobal Projection jacobian of the last bound
/// parametrisation to free parameters
/// @param [in] direction Normalised direction vector
ACTS_DEVICE_FUNC void
reinitializeJacobians(FreeMatrix &transportJacobian, FreeVector &derivatives,
                      BoundToFreeMatrix &jacobianLocalToGlobal,
                      const Vector3D &direction) {
  // Reset the jacobians
  transportJacobian = FreeMatrix::Identity();
  derivatives = FreeVector::Zero();
  jacobianLocalToGlobal = BoundToFreeMatrix::Zero();

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

  jacobianLocalToGlobal(0, eLOC_0) = -sinPhi;
  jacobianLocalToGlobal(0, eLOC_1) = -cosPhi * cosTheta;
  jacobianLocalToGlobal(1, eLOC_0) = cosPhi;
  jacobianLocalToGlobal(1, eLOC_1) = -sinPhi * cosTheta;
  jacobianLocalToGlobal(2, eLOC_1) = sinTheta;
  jacobianLocalToGlobal(3, eT) = 1;
  jacobianLocalToGlobal(4, ePHI) = -sinTheta * sinPhi;
  jacobianLocalToGlobal(4, eTHETA) = cosTheta * cosPhi;
  jacobianLocalToGlobal(5, ePHI) = sinTheta * cosPhi;
  jacobianLocalToGlobal(5, eTHETA) = cosTheta * sinPhi;
  jacobianLocalToGlobal(6, eTHETA) = -sinTheta;
  jacobianLocalToGlobal(7, eQOP) = 1;
}
} // namespace

/// @brief These functions perform the transport of a covariance matrix using
/// given Jacobians. The required data is provided by the stepper object
/// with some additional data. Since this is a purely algebraic problem the
/// calculations are identical for @c StraightLineStepper and @c EigenStepper.
/// As a consequence the methods can be located in a seperate file.
namespace detail {

/// @brief Method for on-demand transport of the covariance to a new frame at
/// current position in parameter space
///
/// @param [in] geoContext The geometry context
/// @param [in, out] covarianceMatrix The covariance matrix of the state
/// @param [in, out] jacobian Full jacobian since the last reset
/// @param [in, out] transportJacobian Global jacobian since the last reset
/// @param [in, out] derivatives Path length derivatives of the free, nominal
/// parameters
/// @param [in, out] jacobianLocalToGlobal Projection jacobian of the last bound
/// parametrisation to free parameters
/// @param [in] parameters Free, nominal parametrisation
/// @param [in] surface is the surface to which the covariance is
///        forwarded to
/// @note No check is done if the position is actually on the surface
template <typename surface_derived_t>
ACTS_DEVICE_FUNC void
covarianceTransport(const GeometryContext &geoContext,
                    BoundSymMatrix &covarianceMatrix, BoundMatrix &jacobian,
                    FreeMatrix &transportJacobian, FreeVector &derivatives,
                    BoundToFreeMatrix &jacobianLocalToGlobal,
                    const FreeVector &parameters, const Surface &surface) {
  // Build the full jacobian (8*8 * 8*6)
  jacobianLocalToGlobal = transportJacobian * jacobianLocalToGlobal;
  const FreeToBoundMatrix jacToLocal = surfaceDerivative<surface_derived_t>(
      geoContext, parameters, jacobianLocalToGlobal, derivatives, surface);
  // Bound to bound jacobian
  jacobian = jacToLocal * jacobianLocalToGlobal;

  // Apply the actual covariance transport
  covarianceMatrix = jacobian * covarianceMatrix * jacobian.transpose();

  // Reinitialize jacobian components
  reinitializeJacobians<surface_derived_t>(geoContext, transportJacobian,
                                           derivatives, jacobianLocalToGlobal,
                                           parameters, surface);
}

/// @brief Method for on-demand transport of the covariance to a new frame at
/// current position in parameter space
///
/// @param [in, out] covarianceMatrix The covariance matrix of the state
/// @param [in, out] jacobian Full jacobian since the last reset
/// @param [in, out] transportJacobian Global jacobian since the last reset
/// @param [in, out] derivatives Path length derivatives of the free, nominal
/// parameters
/// @param [in, out] jacobianLocalToGlobal Projection jacobian of the last bound
/// parametrisation to free parameters
/// @param [in] direction Normalised direction vector
ACTS_DEVICE_FUNC void
covarianceTransport(BoundSymMatrix &covarianceMatrix, BoundMatrix &jacobian,
                    FreeMatrix &transportJacobian, FreeVector &derivatives,
                    BoundToFreeMatrix &jacobianLocalToGlobal,
                    const Vector3D &direction) {
  // Build the full jacobian
  jacobianLocalToGlobal = transportJacobian * jacobianLocalToGlobal;
  const FreeToBoundMatrix jacToLocal =
      surfaceDerivative(direction, jacobianLocalToGlobal, derivatives);
  jacobian = jacToLocal * jacobianLocalToGlobal;

  // Apply the actual covariance transport
  covarianceMatrix = jacobian * covarianceMatrix * jacobian.transpose();

  // Reinitialize jacobian components
  reinitializeJacobians(transportJacobian, derivatives, jacobianLocalToGlobal,
                        direction);
}

/// Create and return the bound state at the current position
///
/// @brief It does not check if the transported state is at the surface, this
/// needs to be guaranteed by the propagator
///
/// @param [in] geoContext The geometry context
/// @param [in, out] covarianceMatrix The covariance matrix of the state
/// @param [in, out] jacobian Full jacobian since the last reset
/// @param [in, out] transportJacobian Global jacobian since the last reset
/// @param [in, out] derivatives Path length derivatives of the free, nominal
/// parameters
/// @param [in, out] jacobianLocalToGlobal Projection jacobian of the last bound
/// parametrisation to free parameters
/// @param [in] parameters Free, nominal parametrisation
/// @param [in] covTransport Decision whether the covariance transport should be
/// performed
/// @param [in] accumulatedPath Propagated distance
/// @param [in] surface Target surface on which the state is represented
///
/// @return A bound state:
///   - the parameters at the surface
///   - the stepwise jacobian towards it (from last bound)
///   - and the path length (from start - for ordering)
template <typename surface_derived_t>
ACTS_DEVICE_FUNC void
boundState(const GeometryContext &geoContext, BoundSymMatrix &covarianceMatrix,
           BoundMatrix &jacobian, FreeMatrix &transportJacobian,
           FreeVector &derivatives, BoundToFreeMatrix &jacobianLocalToGlobal,
           const FreeVector &parameters, bool covTransport,
           const Surface &surface,
           BoundParameters<surface_derived_t> &boundParams) {
  // Covariance transport
  BoundSymMatrix cov = BoundSymMatrix::Zero();
  if (covTransport) {
    // The jacobian, covarianceMatrix, jacobianLocalToGlobal are updated
    covarianceTransport<surface_derived_t>(
        geoContext, covarianceMatrix, jacobian, transportJacobian, derivatives,
        jacobianLocalToGlobal, parameters, surface);
    cov = covarianceMatrix;
  }

  // Create the bound parameters
  const Vector3D &position = parameters.segment<3>(eFreePos0);
  const Vector3D momentum =
      std::abs(1. / parameters[eFreeQOverP]) * parameters.segment<3>(eFreeDir0);
  const double charge = std::copysign(1., parameters[eFreeQOverP]);
  const double time = parameters[eFreeTime];
  boundParams = BoundParameters<surface_derived_t>(
      geoContext, cov, position, momentum, charge, time, &surface);
}

#ifdef __CUDACC__
template <typename surface_derived_t>
__device__ void covarianceTransportOnDevice(
    const GeometryContext &geoContext, BoundSymMatrix &covarianceMatrix,
    BoundMatrix &jacobian, FreeMatrix &transportJacobian,
    FreeVector &derivatives, BoundToFreeMatrix &jacobianLocalToGlobal,
    const FreeVector &parameters, const Surface &surface) {
  __shared__ BoundToFreeMatrix jacToGlobal;
  if (threadIdx.y < eBoundParametersSize) {
    double acc = 0;
    for (int i = 0; i < eFreeParametersSize; i++) {
      acc = transportJacobian(threadIdx.x, i) *
            jacobianLocalToGlobal(i, threadIdx.y);
    }
    jacToGlobal(threadIdx.x, threadIdx.y) = acc;
  }
  __syncthreads();
  // update the jacobianLocalToGlobal
  if (threadIdx.y < eBoundParametersSize) {
    jacobianLocalToGlobal(threadIdx.x, threadIdx.y) =
        jacToGlobal(threadIdx.x, threadIdx.y);
  }
  __syncthreads();

  __shared__ FreeToBoundMatrix jacToLocal;
  // Calculate the jacToLocal with main thread
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    jacToLocal = surfaceDerivative<surface_derived_t>(
        geoContext, parameters, jacobianLocalToGlobal, derivatives, surface);
  }
  __syncthreads();

  // The bound to bound jacobian
  if (threadIdx.x < eBoundParametersSize &&
      threadIdx.y < eBoundParametersSize) {
    double acc = 0;
    for (int i = 0; i < eFreeParametersSize; i++) {
      acc = jacToLocal(threadIdx.x, i) * jacobianLocalToGlobal(i, threadIdx.y);
    }
    jacobian(threadIdx.x, threadIdx.y) = acc;
  }
  __syncthreads();

  __shared__ BoundSymMatrix updatedCovariance;
  // Apply the actual covariance transport
  if (threadIdx.x < eBoundParametersSize &&
      threadIdx.y < eBoundParametersSize) {
    double acc = 0;
    for (int i = 0; i < eFreeParametersSize; i++) {
      for (int j = 0; j < eFreeParametersSize; j++) {
        acc = jacobian(threadIdx.x, i) * covarianceMatrix(i, j) *
              jacobian.transpose()(j, threadIdx.y);
      }
    }
    updatedCovariance(threadIdx.x, threadIdx.y) = acc;
  }
  __syncthreads();
  // update the covariance matrix
  if (threadIdx.x < eBoundParametersSize &&
      threadIdx.y < eBoundParametersSize) {
    covarianceMatrix(threadIdx.x, threadIdx.y) =
        updatedCovariance(threadIdx.x, threadIdx.y);
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    // Reinitialize jacobian components
    reinitializeJacobians<surface_derived_t>(geoContext, transportJacobian,
                                             derivatives, jacobianLocalToGlobal,
                                             parameters, surface);
  }
  __syncthreads();
}

template <typename surface_derived_t>
__device__ void boundStateOnDevice(
    const GeometryContext &geoContext, BoundSymMatrix &covarianceMatrix,
    BoundMatrix &jacobian, FreeMatrix &transportJacobian,
    FreeVector &derivatives, BoundToFreeMatrix &jacobianLocalToGlobal,
    const FreeVector &parameters, bool covTransport, const Surface &surface,
    BoundParameters<surface_derived_t> &boundParams) {
  // Covariance transport
  __shared__ BoundSymMatrix cov;
  // Initialize with multiple threads
  if (threadIdx.x < eBoundParametersSize &&
      threadIdx.y < eBoundParametersSize) {
    cov(threadIdx.x, threadIdx.y) = 0;
  }
  __syncthreads();

  if (covTransport) {
    // update the covariance matrix
    covarianceTransportOnDevice<surface_derived_t>(
        geoContext, covarianceMatrix, jacobian, transportJacobian, derivatives,
        jacobianLocalToGlobal, parameters, surface);
    if (threadIdx.x < eBoundParametersSize &&
        threadIdx.y < eBoundParametersSize) {
      cov(threadIdx.x, threadIdx.y) =
          covarianceMatrix(threadIdx.x, threadIdx.y);
    }
    __syncthreads();
  }

  // Create the bound parameters
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const Vector3D &position = parameters.segment<3>(eFreePos0);
    const Vector3D momentum = std::abs(1. / parameters[eFreeQOverP]) *
                              parameters.segment<3>(eFreeDir0);
    const double charge = std::copysign(1., parameters[eFreeQOverP]);
    const double time = parameters[eFreeTime];

    boundParams = BoundParameters<surface_derived_t>(
        geoContext, cov, position, momentum, charge, time, &surface);
  }
}
#endif

/// Create and return a curvilinear state at the current position
///
/// @brief This creates a curvilinear state.
///
/// @param [in, out] covarianceMatrix The covariance matrix of the state
/// @param [in, out] jacobian Full jacobian since the last reset
/// @param [in, out] transportJacobian Global jacobian since the last reset
/// @param [in, out] derivatives Path length derivatives of the free, nominal
/// parameters
/// @param [in, out] jacobianLocalToGlobal Projection jacobian of the last bound
/// parametrisation to free parameters
/// @param [in] parameters Free, nominal parametrisation
/// @param [in] covTransport Decision whether the covariance transport should be
/// performed
/// @param [in] accumulatedPath Propagated distance
///
/// @return A curvilinear state:
///   - the curvilinear parameters at given position
///   - the stepweise jacobian towards it (from last bound)
///   - and the path length (from start - for ordering)
CurvilinearState ACTS_DEVICE_FUNC curvilinearState(
    BoundSymMatrix &covarianceMatrix, BoundMatrix &jacobian,
    FreeMatrix &transportJacobian, FreeVector &derivatives,
    BoundToFreeMatrix &jacobianLocalToGlobal, const FreeVector &parameters,
    bool covTransport, double accumulatedPath) {
  const Vector3D &direction = parameters.segment<3>(eFreeDir0);

  // Covariance transport
  BoundSymMatrix cov = BoundSymMatrix::Zero();
  if (covTransport) {
    covarianceTransport(covarianceMatrix, jacobian, transportJacobian,
                        derivatives, jacobianLocalToGlobal, direction);
    cov = covarianceMatrix;
  }
  // Create the curvilinear parameters
  const Vector3D &position = parameters.segment<3>(eFreePos0);
  const Vector3D momentum = std::abs(1. / parameters[eFreeQOverP]) * direction;
  const double charge = std::copysign(1., parameters[eFreeQOverP]);
  const double time = parameters[eFreeTime];
  CurvilinearParameters curvilinearParameters(cov, position, momentum, charge,
                                              time);
  // Create the curvilinear state
  return CurvilinearState{std::move(curvilinearParameters), jacobian,
                          accumulatedPath};
}
} // namespace detail
} // namespace Acts
