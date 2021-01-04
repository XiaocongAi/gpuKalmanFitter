// This file is part of the Acts project.
//
// Copyright (C) 2018-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <random>

namespace ActsFatras {

/// Draw random numbers from a Landau distribution.
///
/// Implements the same interface as the standard library distributions.
class LandauDistribution {
public:
  /// Parameter struct that contains all distribution parameters.
  struct param_type {
    /// Parameters must link back to the host distribution.
    using distribution_type = LandauDistribution;

    /// Location parameter.
    ///
    /// @warning This is neither the mean nor the most probable value.
    ActsScalar location = 0.0;
    /// Scale parameter.
    ActsScalar scale = 1.0;

    /// Construct from parameters.
    param_type(ActsScalar location_, ActsScalar scale_)
        : location(location_), scale(scale_) {}
    // Explicitlely defaulted construction and assignment
    param_type() = default;
    param_type(const param_type &) = default;
    param_type(param_type &&) = default;
    param_type &operator=(const param_type &) = default;
    param_type &operator=(param_type &&) = default;

    /// Parameters should be EqualityComparable
    friend bool operator==(const param_type &lhs, const param_type &rhs) {
      return (lhs.location == rhs.location) and (lhs.scale == rhs.scale);
    }
    friend bool operator!=(const param_type &lhs, const param_type &rhs) {
      return not(lhs == rhs);
    }
  };
  /// The type of the generated values.
  using result_type = ActsScalar;

  /// Construct directly from the distribution parameters.
  LandauDistribution(ActsScalar location, ActsScalar scale)
      : m_cfg(location, scale) {}
  /// Construct from a parameter object.
  LandauDistribution(const param_type &cfg) : m_cfg(cfg) {}
  // Explicitlely defaulted construction and assignment
  LandauDistribution() = default;
  LandauDistribution(const LandauDistribution &) = default;
  LandauDistribution(LandauDistribution &&) = default;
  LandauDistribution &operator=(const LandauDistribution &) = default;
  LandauDistribution &operator=(LandauDistribution &&) = default;

  /// Reset any possible internal state. Noop, since there is no internal state.
  void reset() {}
  /// Return the currently configured distribution parameters.
  param_type param() const { return m_cfg; }
  /// Set the distribution parameters.
  void param(const param_type &cfg) { m_cfg = cfg; }

  /// The minimum value the distribution generates.
  result_type min() const {
    return -std::numeric_limits<ActsScalar>::infinity();
  }
  /// The maximum value the distribution generates.
  result_type max() const {
    return std::numeric_limits<ActsScalar>::infinity();
  }

  /// Generate a random number from the configured Landau distribution.
  template <typename Generator> result_type operator()(Generator &generator) {
    return (*this)(generator, m_cfg);
  }
  /// Generate a random number from the given Landau distribution.
  template <typename Generator>
  result_type operator()(Generator &generator, const param_type &params) {
    const auto z = std::uniform_real_distribution<ActsScalar>()(generator);
    return params.location + params.scale * quantile(z);
  }

  /// Provide standard comparison operators
  friend bool operator==(const LandauDistribution &lhs,
                         const LandauDistribution &rhs) {
    return lhs.m_cfg == rhs.m_cfg;
  }
  friend bool operator!=(const LandauDistribution &lhs,
                         const LandauDistribution &rhs) {
    return !(lhs == rhs);
  }

private:
  param_type m_cfg;

  static ActsScalar quantile(ActsScalar z);
};

} // namespace ActsFatras

#include "ActsFatras/Utilities/detail/LandauDistribution.ipp"
