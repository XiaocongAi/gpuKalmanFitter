#pragma once

#include "EventData/TrackParameters.hpp"
#include "Utilities/Math.hpp"
#include <boost/range/adaptors.hpp>
#include <memory>

namespace Acts {

/// @brief Kalman smoother implementation based on Gain matrix formalism
///
/// @tparam parameters_t Type of the track parameters
/// @tparam jacobian_t Type of the Jacobian
template <typename parameters_t> class GainMatrixSmoother {
  using jacobian_t = typename parameters_t::CovarianceMatrix;

public:
  /// @brief Gain Matrix smoother implementation
  ///

  template <typename track_states_t>
  ACTS_DEVICE_FUNC parameters_t *
  operator()(const GeometryContext &gctx,
             track_states_t &filteredStates) const {
    using namespace boost::adaptors;

    using track_state_t = typename track_states_t::value_type;
    using ParVector_t = typename parameters_t::ParametersVector;
    using CovMatrix_t = typename parameters_t::CovarianceMatrix;
    using gain_matrix_t = CovMatrix_t;

    // smoothed parameter vector and covariance matrix
    ParVector_t smoothedPars;
    CovMatrix_t smoothedCov;

    // For the last state: smoothed is filtered - also: switch to next
    track_state_t *prev_ts = &filteredStates[filteredStates.size() - 1];
    prev_ts->parameter.smoothed = prev_ts->parameter.filtered;

    // Smoothing gain matrix
    gain_matrix_t G;

    // Loop and smooth the remaining states
    for (int i = filteredStates.size() - 2; i >= 0; i--) {
      track_state_t &ts = filteredStates[i];

      // The current state
      assert(ts.parameter.filtered);
      assert(ts.parameter.predicted);
      assert(ts.parameter.jacobian);
      assert(ts.parameter.predicted->covariance());
      assert(ts.parameter.filtered->covariance());

      assert(prev_ts->parameter.smoothed);
      assert(prev_ts->parameter.predicted);

      // Gain smoothing matrix
      G = (*ts.parameter.filtered.covariance()) *
          prev_ts->parameter.jacobian.transpose() *
          (BoundSymMatrix)(
              calculateInverse(*prev_ts->parameter.predicted.covariance()));
      // if(G.hasNaN()) {
      // return false;
      //}

      // Calculate the smoothed parameters
      smoothedPars = ts.parameter.filtered.parameters() +
                     G * (prev_ts->parameter.smoothed.parameters() -
                          prev_ts->parameter.predicted.parameters());

      // And the smoothed covariance
      smoothedCov =
          (BoundSymMatrix)(*(ts.parameter.filtered.covariance())) -
          (BoundSymMatrix)(G *
                           (*(prev_ts->parameter.predicted.covariance()) -
                            *(prev_ts->parameter.smoothed.covariance())) *
                           G.transpose());

      // Create smoothed track parameters
      ts.parameter.smoothed = parameters_t(gctx, smoothedCov, smoothedPars,
                                           &(ts.referenceSurface()));

      // Point prev state to current state
      prev_ts = &ts;
    }

    // The result is the pointer to the last smoothed state - for the cache
    return &(prev_ts->parameter.smoothed);
  }
};
} // namespace Acts
