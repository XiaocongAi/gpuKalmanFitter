// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// All functions callable from CUDA code must be qualified with __device__
#ifdef __CUDACC__
#define ACTS_DEVICE_FUNC __host__ __device__
// We need cuda_runtime.h to ensure that that EIGEN_USING_STD_MATH macro
// works properly on the device side
#include <cuda_runtime.h>
#else
#define ACTS_DEVICE_FUNC
#endif

#include "EventData/ChargePolicy.hpp"
#include "EventData/ParameterSet.hpp"
#include "EventData/detail/coordinate_transformations.hpp"
#include "Geometry/GeometryContext.hpp"
#include "Utilities/Definitions.hpp"

#include <type_traits>

namespace Acts {

/// @class SingleTrackParameters
///
/// @brief base class for a single set of track parameters
///
/// This class implements the interface for charged/neutral track parameters for
/// the case that it represents a single set of track parameters
/// (opposed to a list of different sets of track parameters as used by
/// e.g. GSF or multi-track fitters).
///
/// The track parameters and their uncertainty are defined in local reference
/// frame which depends on the associated surface of the track parameters.
///
/// @tparam ChargePolicy type for distinguishing charged and neutral
/// tracks/particles
///         (must be either ChargedPolicy or NeutralPolicy)
template <class ChargePolicy> class SingleTrackParameters {
  static_assert(std::is_same<ChargePolicy, ChargedPolicy>::value or
                    std::is_same<ChargePolicy, NeutralPolicy>::value,
                "ChargePolicy must either be 'Acts::ChargedPolicy' or "
                "'Acts::NeutralPolicy");

public:
  // public typedef's
  using Scalar = BoundParametersScalar;
  /// vector type for stored track parameters
  using ParametersVector = BoundVector;
  /// type of covariance matrix
  using CovarianceMatrix = BoundSymMatrix;

  /// @brief default destructor
  virtual ~SingleTrackParameters() = default;

  /// @brief access position in global coordinate system
  ///
  /// @return 3D vector with global position
  ACTS_DEVICE_FUNC Vector3D position() const { return m_vPosition; }

  /// @brief access momentum in global coordinate system
  /// /// @return 3D vector with global momentum
  ACTS_DEVICE_FUNC Vector3D momentum() const { return m_vMomentum; }

  /// @brief equality operator
  ///
  /// @return @c true of both objects have the same charge policy, parameter
  /// values, position and momentum, otherwise @c false
  ACTS_DEVICE_FUNC bool operator==(const SingleTrackParameters &rhs) const {
    auto casted = dynamic_cast<decltype(this)>(&rhs);
    if (!casted) {
      return false;
    }

    return (m_oChargePolicy == casted->m_oChargePolicy &&
            m_oParameters == casted->m_oParameters &&
            m_vPosition == casted->m_vPosition &&
            m_vMomentum == casted->m_vMomentum);
  }

  /// @brief retrieve electric charge
  ///
  /// @return value of electric charge
  ACTS_DEVICE_FUNC Scalar charge() const { return m_oChargePolicy.getCharge(); }

  /// @brief retrieve time
  ///
  /// @return value of time
  ACTS_DEVICE_FUNC Scalar time() const { return get<eBoundTime>(); }

  /// @brief access to the internally stored ParameterSet
  ///
  /// @return ParameterSet object holding parameter values and their covariance
  /// matrix
  ACTS_DEVICE_FUNC const FullParameterSet &getParameterSet() const {
    return m_oParameters;
  }

  /// @brief access associated surface defining the coordinate system for track
  ///        parameters and their covariance
  ///
  /// @return associated surface
  ACTS_DEVICE_FUNC virtual const Surface &referenceSurface() const = 0;

  /// @brief access covariance matrix of track parameters
  ///
  /// @note The ownership of the covariance matrix is @b not transferred with
  /// this call.
  ///
  /// @sa ParameterSet::getCovariance
  ACTS_DEVICE_FUNC const CovarianceMatrix *covariance() const {
    return getParameterSet().getCovariance();
  }

  /// @brief access track parameters
  ///
  /// @return Eigen vector of dimension Acts::eBoundParametersSize with values
  /// of the track parameters
  ///         (in the order as defined by the ParID_t enumeration)
  ACTS_DEVICE_FUNC ParametersVector parameters() const {
    return getParameterSet().getParameters();
  }

  /// @brief access track parameter
  ///
  /// @tparam par identifier of track parameter which is to be retrieved
  ///
  /// @return value of the requested track parameter
  ///
  /// @sa ParameterSet::get
  template <BoundParametersIndices par> ACTS_DEVICE_FUNC Scalar get() const {
    return getParameterSet().template getParameter<par>();
  }

  /// @brief access track parameter uncertainty
  ///
  /// @tparam par identifier of track parameter which is to be retrieved
  ///
  /// @return value of the requested track parameter uncertainty
  template <BoundParametersIndices par>
  ACTS_DEVICE_FUNC Scalar uncertainty() const {
    return getParameterSet().template getUncertainty<par>();
  }

  /// @brief convenience method to retrieve transverse momentum
  ACTS_DEVICE_FUNC Scalar pT() const { return VectorHelpers::perp(momentum()); }

  /// @brief convenience method to retrieve pseudorapidity
  ACTS_DEVICE_FUNC Scalar eta() const { return VectorHelpers::eta(momentum()); }

  ACTS_DEVICE_FUNC FullParameterSet &getParameterSet() { return m_oParameters; }

protected:
  /// @brief standard constructor for track parameters of charged particles
  ///
  /// @param cov unique pointer to covariance matrix (nullptr is accepted)
  /// @param parValues vector with parameter values
  /// @param position 3D vector with global position
  /// @param momentum 3D vector with global momentum
  template <typename T = ChargePolicy,
            std::enable_if_t<std::is_same<T, ChargedPolicy>::value, int> = 0>
  ACTS_DEVICE_FUNC SingleTrackParameters(const CovarianceMatrix &cov,
                                         const ParametersVector &parValues,
                                         const Vector3D &position,
                                         const Vector3D &momentum)
      : m_oChargePolicy(
            detail::coordinate_transformation::parameters2charge(parValues)),
        m_oParameters(std::move(cov), parValues), m_vPosition(position),
        m_vMomentum(momentum) {}

  /// @brief standard constructor for track parameters of neutral particles
  ///
  /// @param cov unique pointer to covariance matrix (nullptr is accepted)
  /// @param parValues vector with parameter values
  /// @param position 3D vector with global position
  /// @param momentum 3D vector with global momentum
  template <typename T = ChargePolicy,
            std::enable_if_t<std::is_same<T, NeutralPolicy>::value, int> = 0>
  ACTS_DEVICE_FUNC SingleTrackParameters(const CovarianceMatrix &cov,
                                         const ParametersVector &parValues,
                                         const Vector3D &position,
                                         const Vector3D &momentum)
      : m_oChargePolicy(), m_oParameters(std::move(cov), parValues),
        m_vPosition(position), m_vMomentum(momentum) {}

  /// @brief default copy constructor
  SingleTrackParameters(const SingleTrackParameters<ChargePolicy> &copy) =
      default;

  /// @brief default move constructor
  SingleTrackParameters(SingleTrackParameters<ChargePolicy> &&copy) = default;

  /// @brief copy assignment operator
  ///
  /// @param rhs object to be copied
  SingleTrackParameters<ChargePolicy> &ACTS_DEVICE_FUNC
  operator=(const SingleTrackParameters<ChargePolicy> &rhs) {
    // check for self-assignment
    if (this != &rhs) {
      m_oChargePolicy = rhs.m_oChargePolicy;
      m_oParameters = rhs.m_oParameters;
      m_vPosition = rhs.m_vPosition;
      m_vMomentum = rhs.m_vMomentum;
    }

    return *this;
  }

  /// @brief move assignment operator
  ///
  /// @param rhs object to be movied into `*this`
  SingleTrackParameters<ChargePolicy> &ACTS_DEVICE_FUNC
  operator=(SingleTrackParameters<ChargePolicy> &&rhs) {
    // check for self-assignment
    if (this != &rhs) {
      m_oChargePolicy = std::move(rhs.m_oChargePolicy);
      m_oParameters = std::move(rhs.m_oParameters);
      m_vPosition = std::move(rhs.m_vPosition);
      m_vMomentum = std::move(rhs.m_vMomentum);
    }

    return *this;
  }

  /// @brief update global momentum from current parameter values
  ///
  ///
  /// @param[in] gctx is the Context object that is forwarded to the surface
  ///            for local to global coordinate transformation
  ///
  /// @note This function is triggered when called with an argument of a type
  ///       different from Acts::local_parameter
  template <typename T>
  ACTS_DEVICE_FUNC void
  updateGlobalCoordinates(const GeometryContext & /*gctx*/,
                          const T & /*unused*/) {
    m_vMomentum = detail::coordinate_transformation::parameters2globalMomentum(
        getParameterSet().getParameters());
  }

  /// @brief update global position from current parameter values
  ///
  /// @note This function is triggered when called with an argument of a type
  /// Acts::local_parameter
  ACTS_DEVICE_FUNC void
  updateGlobalCoordinates(const GeometryContext &gctx,
                          const local_parameter & /*unused*/) {
    m_vPosition = detail::coordinate_transformation::parameters2globalPosition(
        gctx, getParameterSet().getParameters(), this->referenceSurface());
  }

  ChargePolicy m_oChargePolicy;   ///< charge policy object distinguishing
                                  /// between charged and neutral tracks
  FullParameterSet m_oParameters; ///< ParameterSet object holding the
                                  /// parameter values and covariance matrix
  Vector3D m_vPosition;           ///< 3D vector with global position
  Vector3D m_vMomentum;           ///< 3D vector with global momentum
};

} // namespace Acts
