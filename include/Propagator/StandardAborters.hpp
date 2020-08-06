// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Propagator/ConstrainedStep.hpp"
#include "Surfaces/BoundaryCheck.hpp"
#include "Surfaces/Surface.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Intersection.hpp"

#include <limits>
#include <sstream>
#include <string>

namespace Acts {

/// This is the condition that the pathLimit has been reached
struct PathLimitReached {
  /// Boolean switch for Loop protection
  double internalLimit = std::numeric_limits<double>::max();

  /// boolean operator for abort condition without using the result
  ///
  /// @tparam propagator_state_t Type of the propagator state
  /// @tparam stepper_t Type of the stepper
  ///
  /// @param [in,out] state The propagation state object
  template <typename propagator_state_t, typename stepper_t>
  bool operator()(propagator_state_t &state,
                  const stepper_t & /*unused*/) const {
    if (state.navigation.targetReached) {
      return true;
    }
    // Check if the maximum allowed step size has to be updated
    double distance = state.stepping.navDir * std::abs(internalLimit) -
                      state.stepping.pathAccumulated;
    double tolerance = state.options.targetTolerance;
    state.stepping.stepSize.update(distance, ConstrainedStep::aborter);
    bool limitReached = (distance * distance < tolerance * tolerance);
    if (limitReached) {
      // reaching the target means navigation break
      state.navigation.targetReached = true;
    }
    // path limit check
    return limitReached;
  }
};

/// This is the condition that the Surface has been reached
/// it then triggers an propagation abort of the propagation
struct SurfaceReached {
  SurfaceReached() = default;

  /// boolean operator for abort condition without using the result
  ///
  /// @tparam propagator_state_t Type of the propagator state
  /// @tparam stepper_t Type of the stepper
  ///
  /// @param [in,out] state The propagation state object
  /// @param [in] stepper Stepper used for propagation
  template <typename propagator_state_t, typename stepper_t>
  bool operator()(propagator_state_t &state, const stepper_t &stepper) const {
    return (*this)(state, stepper, *state.navigation.targetSurface);
  }

  /// boolean operator for abort condition without using the result
  ///
  /// @tparam propagator_state_t Type of the propagator state
  /// @tparam stepper_t Type of the stepper
  ///
  /// @param [in,out] state The propagation state object
  /// @param [in] stepper Stepper used for the progation
  /// @param [in] targetSurface The target surface
  template <typename propagator_state_t, typename stepper_t>
  bool operator()(propagator_state_t &state, const stepper_t &stepper,
                  const Surface &targetSurface) const {
    if (state.navigation.targetReached) {
      return true;
    }
    // Check if the cache filled the currentSurface - or if we are on the
    // surface
    if ((state.navigation.currentSurface &&
         state.navigation.currentSurface == &targetSurface)) {
      // reaching the target calls a navigation break
      state.navigation.targetReached = true;
      return true;
    }
    // Calculate the distance to the surface
    const double tolerance = state.options.targetTolerance;
    const auto sIntersection = targetSurface.intersect(
        state.geoContext, stepper.position(state.stepping),
        state.stepping.navDir * stepper.direction(state.stepping), true);

    // The target is reached
    bool targetReached =
        (sIntersection.intersection.status == Intersection::Status::onSurface);
    double distance = sIntersection.intersection.pathLength;

    // Return true if you fall below tolerance
    if (targetReached) {
      // assigning the currentSurface
      state.navigation.currentSurface = &targetSurface;
      // reaching the target calls a navigation break
      state.navigation.targetReached = true;
    } else {
      // Target is not reached, update the step size
      const double overstepLimit = stepper.overstepLimit(state.stepping);
      // Check the alternative solution
      if (distance < overstepLimit and sIntersection.alternative) {
        // Update the distance to the alternative solution
        distance = sIntersection.alternative.pathLength;
      }
      state.stepping.stepSize.update(state.stepping.navDir * distance,
                                     ConstrainedStep::aborter);
    }
    // path limit check
    return targetReached;
  }
};

} // namespace Acts
