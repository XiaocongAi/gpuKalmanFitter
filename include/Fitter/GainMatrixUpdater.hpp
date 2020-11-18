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
#include "Utilities/Helpers.hpp"

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

    ParVector_t filtered_parameters;
    CovMatrix_t filtered_covariance;

    // The source link
    const auto &sl = trackState.measurement.uncalibrated;

    // Take the projector (measurement mapping function)
    const auto &H = sl.projector();
    meas_cov_t cov = H * predicted_covariance * H.transpose() + sl.covariance();
    meas_cov_t covInv = get2DMatrixInverse(cov);
    // The Kalman gain matrix
    const ActsMatrixD<eBoundParametersSize, measdim> K =
        predicted_covariance * H.transpose() * covInv;

    // filtered new parameters after update
    const ParVector_t gain = K * sl.residual(predicted);
    filtered_parameters = predicted.parameters() + gain;

    // @todo use multiple threads for this
    // printf("K = (%f\n, %f\n, %f\n, %f\n, %f\n, %f)\n", K(0,0), K(0,1),
    // K(1,0), K(1,1), K(2,0), K(2,1), K(3,0), K(3,1), K(4,0),
    // K(4,1),K(5,0),K(5,1));
    const CovMatrix_t KH = K * H;
    const CovMatrix_t C = CovMatrix_t::Identity() - KH;
    // printf("C = (%f\n, %f\n, %f\n, %f\n, %f\n, %f)\n", C(0,0), C(0,1),
    // C(1,0), C(1,1), C(2,0), C(2,1), C(3,0), C(3,1), C(4,0),
    // C(4,1),C(5,0),C(5,1));

    // updated covariance after filtering
    filtered_covariance = C * predicted_covariance;

    // Create new filtered parameters and covariance
    parameters_t filtered(gctx, std::move(filtered_covariance),
                          filtered_parameters, &sl.referenceSurface());

    //    // calculate the chi2
    //    // chi2 = r^T * R^-1 * r
    //    // r is the residual of the filtered state
    //    // R is the covariance matrix of the filtered residual
    //    meas_cov_t R = (meas_cov_t::Identity() - H * K) * sl.covariance();
    //    meas_par_t residual = sl.residual(filtered);
    //    trackState.parameter.chi2 =
    //        (residual.transpose() *
    //         get2DMatrixInverse(R) *
    //         residual)
    //            .eval()(0, 0);
    //
    trackState.parameter.filtered = filtered;

    // always succeed, no outlier logic yet
    return true;
  }

#ifdef __CUDACC__
  // The updater with multiple threads on GPU
  template <typename track_state_t>
  __device__ bool updateOnDevice(const GeometryContext &gctx,
                                 track_state_t &trackState) const {
    const bool IS_MAIN_THREAD = threadIdx.x == 0 && threadIdx.y == 0;

    // printf("Invoked GainMatrixUpdater\n");
    using parameters_t = typename track_state_t::Parameters;
    using source_link_t = typename track_state_t::SourceLink;

    using ParVector_t = typename parameters_t::ParametersVector;
    using CovMatrix_t = typename parameters_t::CovarianceMatrix;

    using projector_t = typename source_link_t::projector_t;
    using meas_par_t = typename source_link_t::meas_par_t;
    using meas_cov_t = typename source_link_t::meas_cov_t;

    constexpr size_t measdim = meas_par_t::RowsAtCompileTime;

    // The source link
    const auto &sl = trackState.measurement.uncalibrated;
    // Take the projector (measurement mapping function)
    const auto &H = sl.projector();

    // read-only prediction handle
    const parameters_t &predicted = trackState.parameter.predicted;
    const ParVector_t &predicted_parameters = predicted.parameters();
    const CovMatrix_t &predicted_covariance = *predicted.covariance();

    // The filtered parameter to be written into
    // parameters_t &filtered = trackState.parameter.filtered;
    // ParVector_t &filtered_parameters = filtered.parameters();
    // CovMatrix_t &filtered_covariance = *filtered.covariance();

    __shared__ ActsMatrixD<measdim, eBoundParametersSize> HP;
    __shared__ meas_cov_t cov;
    __shared__ meas_cov_t covInv;
    __shared__ ActsMatrixD<eBoundParametersSize, measdim> K;
    __shared__ CovMatrix_t C;

    // Use multiple threads for the calculation of cov = H *
    // predicted_covariance * H.transpose() + sl.covariance();
    if (threadIdx.x < 2 and threadIdx.y < eBoundParametersSize) {
      double acc = 0;
      for (unsigned int i = 0; i < eBoundParametersSize; i++) {
        acc += H(threadIdx.x, i) * predicted_covariance(i, threadIdx.y);
      }
      HP(threadIdx.x, threadIdx.y) = acc;
    }
    __syncthreads();
    if (threadIdx.x < 2 and threadIdx.y < 2) {
      double acc = 0;
      for (unsigned int i = 0; i < eBoundParametersSize; i++) {
        acc += HP(threadIdx.x, i) * H(threadIdx.y, i);
      }
      cov(threadIdx.x, threadIdx.y) =
          acc + sl.covariance()(threadIdx.x, threadIdx.y);
    }
    __syncthreads();

    // Calculate the inverse with the main thread
    if (IS_MAIN_THREAD) {
      covInv = get2DMatrixInverse(cov);
      // printf("covInv = (%f, %f, %f, %f)\n", covInv(0,0), covInv(0,1),
      // covInv(1,0), covInv(1,1));
    }
    __syncthreads();

    // The Kalman gain matrix
    // const ActsMatrixD<eBoundParametersSize, measdim> K =
    //   predicted_covariance * H.transpose() * covInv;
    if (threadIdx.x < eBoundParametersSize and threadIdx.y < 2) {
      double acc = 0;
      for (unsigned int i = 0; i < eBoundParametersSize; i++) {
        for (unsigned int j = 0; j < 2; j++) {
          acc += predicted_covariance(threadIdx.x, i) * H(j, i) *
                 covInv(j, threadIdx.y);
        }
      }
      K(threadIdx.x, threadIdx.y) = acc;
    }
    __syncthreads();

    // if (IS_MAIN_THREAD) {
    // printf("K = (%f\n, %f\n, %f\n, %f\n, %f\n, %f)\n", K(0,0), K(0,1),
    // K(1,0), K(1,1), K(2,0), K(2,1), K(3,0), K(3,1), K(4,0),
    // K(4,1),K(5,0),K(5,1));
    //}
    //__syncthreads();

    // @todo use multiple threads for this
    if (threadIdx.x < eBoundParametersSize and
        threadIdx.y < eBoundParametersSize) {
      double acc = 0;
      for (unsigned int i = 0; i < 2; i++) {
        acc += K(threadIdx.x, i) * H(i, threadIdx.y);
      }
      double iden = (threadIdx.x == threadIdx.y) ? 1.0 : 0;
      C(threadIdx.x, threadIdx.y) = iden - acc;
    }
    __syncthreads();

    // if (IS_MAIN_THREAD) {
    // printf("C = (%f\n, %f\n, %f\n, %f\n, %f\n, %f)\n", C(0,0), C(0,1),
    // C(1,0), C(1,1), C(2,0), C(2,1), C(3,0), C(3,1), C(4,0),
    // C(4,1),C(5,0),C(5,1));
    //}
    //__syncthreads();

    if (IS_MAIN_THREAD) {
      ParVector_t filtered_parameters;
      CovMatrix_t filtered_covariance;

      // filtered new parameters after update
      const ParVector_t gain = K * sl.residual(predicted);
      filtered_parameters = predicted.parameters() + gain;

      // updated covariance after filtering
      filtered_covariance = C * predicted_covariance;

      // Create new filtered parameters and covariance
      parameters_t filtered(gctx, std::move(filtered_covariance),
                            std::move(filtered_parameters),
                            &sl.referenceSurface());
      trackState.parameter.filtered = filtered;
    }
    __syncthreads();

    // always succeed, no outlier logic yet
    return true;
  }
#endif
};

} // namespace Acts
