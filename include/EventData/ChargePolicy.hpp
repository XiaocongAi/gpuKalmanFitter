// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
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

namespace Acts {

/// @class ChargedPolicy
///
/// @brief policy class for charged particles/tracks
///
/// This type is meant to be used as template parameter to the
/// SingleTrackParameters class
/// and its derived classes in order to provide a distinction between charged
/// and
/// neutral
/// track (parameters) at the C++ type level. This allows other class to employ
/// optimized
/// algorithms for either case by using template specializations.
class ChargedPolicy {
public:
  /// @brief constructor with given charge
  ///
  /// @param charge electric charge of particle/track (parameters)
  ACTS_DEVICE_FUNC ChargedPolicy(ActsScalar charge) : m_dCharge(charge) {}

  /// @brief equality operator
  ///
  /// @return @c true if rhs has the same charge, otherwise @c false
  ACTS_DEVICE_FUNC bool operator==(const ChargedPolicy &rhs) const {
    return m_dCharge == rhs.m_dCharge;
  }

  /// @brief inequality operator
  ///
  /// @return @c true if rhs has a different charge, otherwise @c false
  ACTS_DEVICE_FUNC bool operator!=(const ChargedPolicy &rhs) const {
    return !(*this == rhs);
  }

  /// @brief retrieve stored value of the electric charge
  ///
  /// @return value for charge
  ACTS_DEVICE_FUNC ActsScalar getCharge() const { return m_dCharge; }

  /// @brief sets charge
  ///
  /// @param charge new value for the electric charge
  ACTS_DEVICE_FUNC void setCharge(ActsScalar charge) { m_dCharge = charge; }

  /// @brief flip sign of electric charge
  ACTS_DEVICE_FUNC void flipSign() { m_dCharge *= -1.; }

private:
  ActsScalar m_dCharge; ///< value of electric charge
};

/// @class NeutralPolicy
///
/// @brief policy class for neutral particles/tracks
///
/// This type is meant to be used as template parameter to the
/// SingleTrackParameters class
/// and its derived classes in order to provide a distinction between charged
/// and
/// neutral
/// track (parameters) at the C++ type level. This allows other class to employ
/// optimized
/// algorithms for either case by using template specializations.
class NeutralPolicy {
public:
  /// @brief equality operator
  ///
  /// @return always @c true
  bool operator==(const NeutralPolicy & /*other*/) const { return true; }

  /// @brief inequality operator
  ///
  /// @return always @c false
  bool operator!=(const NeutralPolicy &rhs) const { return !(*this == rhs); }

  /// @brief get electric charge
  ///
  /// @return always 0
  ActsScalar getCharge() const { return 0.; }
};
} // namespace Acts
