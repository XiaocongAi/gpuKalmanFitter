// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EventData/TrackParameters.hpp"
#include "Fitter/detail/VoidKalmanComponents.hpp"
#include "Utilities/Definitions.hpp"

#include <memory>

namespace Acts {

/// @brief Update step of Kalman Filter using gain matrix formalism
class GainMatrixUpdater {
public:
  /// @brief Public call operator for the boost visitor pattern
  ///
  /// @tparam track_state_t Type of the track state for the update
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param trackState the measured track state
  /// @param direction the navigation direction
  ///
  /// @return Bool indicating whether this update was 'successful'
  /// @note Non-'successful' updates could be holes or outliers,
  ///       which need to be treated differently in calling code.
  template <typename track_state_t>
  ACTS_DEVICE_FUNC bool operator()(const GeometryContext &gctx,
                                   track_state_t &trackState) const {
    // printf("Invoked GainMatrixUpdater\n");
    using parameters_t = typename track_state_t::Parameters;
    using source_link_t = typename track_state_t::SourceLink;

    using CovMatrix_t = typename parameters_t::CovarianceMatrix;
    using ParVector_t = typename parameters_t::ParametersVector;

    using projector_t = typename source_link_t::projector_t;
    using meas_par_t = typename source_link_t::meas_par_t;
    using meas_cov_t = typename source_link_t::meas_cov_t;

    constexpr size_t measdim = meas_par_t::RowsAtCompileTime;

    // read-only prediction handle
    const parameters_t &predicted = trackState.parameter.predicted;
    const CovMatrix_t &predicted_covariance = *predicted.covariance();

    // ACTS_VERBOSE("Predicted parameters: " << predicted.transpose());
    // ACTS_VERBOSE("Predicted covariance:\n" << predicted_covariance);

    ParVector_t filtered_parameters;
    CovMatrix_t filtered_covariance;

    // The source link
    const auto &sl = trackState.measurement.uncalibrated;

    // Take the projector (measurement mapping function)
    const auto &H = sl.projector();

    // The Kalman gain matrix
    const ActsMatrixD<eBoundParametersSize, measdim> K =
        predicted_covariance * H.transpose() *
        (H * predicted_covariance * H.transpose() + sl.covariance()).inverse();

    // filtered new parameters after update
    filtered_parameters = predicted.parameters() + K * sl.residual(predicted);

    // updated covariance after filtering
    filtered_covariance =
        (CovMatrix_t::Identity() - K * H) * predicted_covariance;

    // Create new filtered parameters and covariance
    parameters_t filtered(gctx, std::move(filtered_covariance),
                          filtered_parameters, &predicted.referenceSurface());

    // calculate the chi2
    // chi2 = r^T * R^-1 * r
    // r is the residual of the filtered state
    // R is the covariance matrix of the filtered residual
    meas_par_t residual = sl.residual(filtered);
    trackState.parameter.chi2 =
        (residual.transpose() *
         ((meas_cov_t::Identity() - H * K) * sl.covariance()).inverse() *
         residual)
            .eval()(0, 0);

    trackState.parameter.filtered = filtered;

    // always succeed, no outlier logic yet
    return true;
  }
};

} // namespace Acts
