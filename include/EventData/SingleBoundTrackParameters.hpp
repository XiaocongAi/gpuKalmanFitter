// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EventData/SingleTrackParameters.hpp"
#include "Geometry/GeometryContext.hpp"
#include "Surfaces/PlaneSurface.hpp"

namespace Acts {

/// @brief Charged and Neutrial Track Parameterisation classes bound to
/// to a reference surface. This is a single-component representation
///
///
/// Bound track parameters are delegating the transformation of local to global
/// coordinate transformations to the reference surface Surface and thus need
/// at contruction a Context object
///
/// @note This class holds shared ownership on the surface it is associated
///       to.
template <class ChargePolicy,
          typename surface_derived_t = PlaneSurface<InfiniteBounds>>
class SingleBoundTrackParameters : public SingleTrackParameters<ChargePolicy> {
public:
  using Scalar = BoundParametersScalar;
  using ParametersVector = BoundVector;
  using CovarianceMatrix = BoundSymMatrix;
  using ReferenceSurfaceType = surface_derived_t;

  /// @brief Construct without parameters (to be used in TrackState)
  /// @To to removed
  template <typename T = ChargePolicy,
            std::enable_if_t<std::is_same<T, ChargedPolicy>::value, int> = 0>
  ACTS_DEVICE_FUNC SingleBoundTrackParameters()
      : SingleTrackParameters<ChargePolicy>(
            CovarianceMatrix::Zero(), ParametersVector::Zero(),
            Vector3D(0, 0, 0), Vector3D(0, 0, 0)) {}

  /// @brief Constructor of track parameters bound to a surface
  /// This is the constructor from global parameters, enabled only
  /// for charged representations.
  ///
  /// The transformations declared in the coordinate_transformation
  /// yield the global parameters and momentum representation
  /// @param[in] gctx is the Context object that is forwarded to the surface
  ///            for local to global coordinate transformation
  /// @param[in] cov The covaraniance matrix (optional, can be nullptr)
  ///            it is given in the measurement frame
  /// @param[in] parValues The parameter vector
  /// @param[in] surface The reference surface the parameters are bound to
  template <typename T = ChargePolicy,
            std::enable_if_t<std::is_same<T, ChargedPolicy>::value, int> = 0>
  ACTS_DEVICE_FUNC SingleBoundTrackParameters(const GeometryContext &gctx,
                                              const CovarianceMatrix &cov,
                                              const ParametersVector &parValues,
                                              const Surface *surface)
      : SingleTrackParameters<ChargePolicy>(
            std::move(cov), parValues,
            detail::coordinate_transformation::parameters2globalPosition<
                ReferenceSurfaceType>(gctx, parValues, *surface),
            detail::coordinate_transformation::parameters2globalMomentum(
                parValues)),
        m_pSurface(surface) {
    assert(m_pSurface);
  }

  /// @brief Constructor of track parameters bound to a surface
  /// This is the constructor from global parameters, enabled only
  /// for charged representations.
  ///
  /// The transformations declared in the coordinate_transformation
  /// yield the local parameters
  ///
  /// @param[in] gctx is the Context object that is forwarded to the surface
  ///            for local to global coordinate transformation
  /// @param[in] cov The covaraniance matrix (optional, can be nullptr)
  ///            it is given in the curvilinear frame
  /// @param[in] position The global position of the track parameterisation
  /// @param[in] momentum The global momentum of the track parameterisation
  /// @param[in] dCharge The charge of the particle track parameterisation
  /// @param[in] dTime The time component of the parameters
  /// @param[in] surface The reference surface the parameters are bound to
  template <typename T = ChargePolicy,
            std::enable_if_t<std::is_same<T, ChargedPolicy>::value, int> = 0>
  ACTS_DEVICE_FUNC SingleBoundTrackParameters(const GeometryContext &gctx,
                                              const CovarianceMatrix &cov,
                                              const Vector3D &position,
                                              const Vector3D &momentum,
                                              Scalar dCharge, Scalar dTime,
                                              const Surface *surface)
      : SingleTrackParameters<ChargePolicy>(
            std::move(cov),
            detail::coordinate_transformation::global2parameters<
                ReferenceSurfaceType>(gctx, position, momentum, dCharge, dTime,
                                      *surface),
            position, momentum),
        m_pSurface(std::move(surface)) {
    assert(m_pSurface);
  }

  /// @brief Constructor of track parameters bound to a surface
  /// This is the constructor from global parameters, enabled only
  /// for neutral representations.
  ///
  /// The transformations declared in the coordinate_transformation
  /// yield the global parameters and momentum representation
  ///
  /// @param[in] gctx is the Context object that is forwarded to the surface
  ///            for local to global coordinate transformation
  /// @param[in] cov The covaraniance matrix (optional, can be nullptr)
  ///            it is given in the measurement frame
  /// @param[in] parValues The parameter vector
  /// @param[in] surface The reference surface the parameters are bound to
  template <typename T = ChargePolicy,
            std::enable_if_t<std::is_same<T, NeutralPolicy>::value, int> = 0>
  ACTS_DEVICE_FUNC SingleBoundTrackParameters(const GeometryContext &gctx,
                                              const CovarianceMatrix &cov,
                                              const ParametersVector &parValues,
                                              const Surface *surface)
      : SingleTrackParameters<ChargePolicy>(
            std::move(cov), parValues,
            detail::coordinate_transformation::parameters2globalPosition<
                ReferenceSurfaceType>(gctx, parValues, *surface),
            detail::coordinate_transformation::parameters2globalMomentum(
                parValues)),
        m_pSurface(std::move(surface)) {
    assert(m_pSurface);
  }

  /// @brief Constructor of track parameters bound to a surface
  /// This is the constructor from global parameters, enabled only
  /// for neutral representations.
  ///
  /// The transformations declared in the coordinate_transformation
  /// yield the local parameters
  ///
  ///
  /// @param[in] gctx is the Context object that is forwarded to the surface
  ///            for local to global coordinate transformation
  /// @param[in] cov The covaraniance matrix (optional, can be nullptr)
  ///            it is given in the curvilinear frame
  /// @param[in] position The global position of the track parameterisation
  /// @param[in] momentum The global momentum of the track parameterisation
  /// @param[in] dCharge The charge of the particle track parameterisation
  /// @param[in] surface The reference surface the parameters are bound to
  template <typename T = ChargePolicy,
            std::enable_if_t<std::is_same<T, NeutralPolicy>::value, int> = 0>
  ACTS_DEVICE_FUNC
  SingleBoundTrackParameters(const GeometryContext &gctx,
                             const CovarianceMatrix &cov,
                             const Vector3D &position, const Vector3D &momentum,
                             Scalar dTime, const Surface *surface)
      : SingleTrackParameters<ChargePolicy>(
            std::move(cov),
            detail::coordinate_transformation::global2parameters<
                ReferenceSurfaceType>(gctx, position, momentum, 0, dTime,
                                      *surface),
            position, momentum),
        m_pSurface(std::move(surface)) {}

  /// @brief copy constructor  - charged/neutral
  /// @param[in] copy The source parameters
  ACTS_DEVICE_FUNC SingleBoundTrackParameters(
      const SingleBoundTrackParameters<ChargePolicy> &copy)
      : SingleTrackParameters<ChargePolicy>(copy) {
    m_pSurface = copy.m_pSurface;
  }

  /// @brief move constructor - charged/neutral
  /// @param[in] other The source parameters
  ACTS_DEVICE_FUNC
  SingleBoundTrackParameters(SingleBoundTrackParameters<ChargePolicy> &&other)
      : SingleTrackParameters<ChargePolicy>(std::move(other)),
        m_pSurface(std::move(other.m_pSurface)) {}

  /// @brief desctructor - charged/neutral
  /// checks if the surface is free and in such a case deletes it
  ~SingleBoundTrackParameters() = default;

  /// @brief copy assignment operator - charged/neutral
  ACTS_DEVICE_FUNC SingleBoundTrackParameters<ChargePolicy> &
  operator=(const SingleBoundTrackParameters<ChargePolicy> &rhs) {
    // check for self-assignment
    if (this != &rhs) {
      SingleTrackParameters<ChargePolicy>::operator=(rhs);
      m_pSurface = rhs.m_pSurface;
    }
    return *this;
  }

  /// @brief move assignment operator - charged/neutral
  /// checks if the surface is free and in such a case delete-clones it
  ACTS_DEVICE_FUNC SingleBoundTrackParameters<ChargePolicy> &
  operator=(SingleBoundTrackParameters<ChargePolicy> &&rhs) {
    // check for self-assignment
    if (this != &rhs) {
      SingleTrackParameters<ChargePolicy>::operator=(std::move(rhs));
      m_pSurface = std::move(rhs.m_pSurface);
    }

    return *this;
  }

  /// @brief set method for parameter updates
  /// obviously only allowed on non-const objects
  //
  /// @param[in] gctx is the Context object that is forwarded to the surface
  ///            for local to global coordinate transformation
  template <ParID_t par>
  ACTS_DEVICE_FUNC void set(const GeometryContext &gctx, Scalar newValue) {
    this->getParameterSet().template setParameter<par>(newValue);
    this->updateGlobalCoordinates(gctx, BoundParameterType<par>());
  }

  /// @brief access method to the reference surface
  ACTS_DEVICE_FUNC const Surface &referenceSurface() const final {
    return *m_pSurface;
  }

private:
  const Surface *m_pSurface = nullptr;
};

} // namespace Acts
