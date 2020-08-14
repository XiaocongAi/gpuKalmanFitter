// This file is part of the Acts project.
//
// Copyright (C) 2016-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Geometry/GeometryContext.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include <stdexcept>
#include <string>

namespace Acts {

/// A minimal Pixel Source link class
///
///
class PixelSourceLink {
public:
  using projector_t =
      ActsMatrix<BoundParametersScalar, 2, eBoundParametersSize>;
  using meas_par_t = ActsVector<FreeParametersScalar, 2>;
  using meas_cov_t = ActsMatrix<FreeParametersScalar, 2, 2>;

  ACTS_DEVICE_FUNC PixelSourceLink(const meas_par_t &values,
                                   const meas_cov_t &cov,
                                   const Surface *surface)
      : m_values(values), m_cov(cov), m_surface(surface) {}
  /// Must be default_constructible to satisfy SourceLinkConcept.
  PixelSourceLink() = default;
  PixelSourceLink(PixelSourceLink &&) = default;
  PixelSourceLink(const PixelSourceLink &) = default;
  PixelSourceLink &operator=(PixelSourceLink &&) = default;
  PixelSourceLink &operator=(const PixelSourceLink &) = default;

  constexpr const meas_par_t &localPosition() const { return m_values; }

  Vector3D globalPosition(const GeometryContext &gctx) const {
    Vector3D global(0, 0, 0);
    Vector3D mom(1, 1, 1);
    m_surface->localToGlobal(gctx, m_values, mom, global);
    return global;
  }

  ACTS_DEVICE_FUNC constexpr const meas_cov_t &covariance() const {
    return m_cov;
  }

  ACTS_DEVICE_FUNC constexpr const Surface &referenceSurface() const {
    return *m_surface;
  }

  ACTS_DEVICE_FUNC projector_t projector() const {
    return projector_t::Identity();
  }

  template <typename parameters_t>
  ACTS_DEVICE_FUNC meas_par_t residual(const parameters_t &par) const {
    meas_par_t residual = meas_par_t::Zero();
    residual[0] = m_values[0] - par.parameters()[0];
    residual[1] = m_values[1] - par.parameters()[1];
    return residual;
  }

private:
  meas_par_t m_values;
  meas_cov_t m_cov;
  // need to store pointers to make the object copyable
  const Surface *m_surface = nullptr;
};
} // namespace Acts
