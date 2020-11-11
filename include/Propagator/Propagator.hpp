// This file is part of the Acts project.
//
// Copyright (C) 2016-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Geometry/GeometryContext.hpp"
#include "MagneticField/MagneticFieldContext.hpp"

#include "EventData/TrackParameters.hpp"
#include "Propagator/DirectNavigator.hpp"
#include "Propagator/StandardAborters.hpp"
#include "Utilities/Definitions.hpp"

#include <Eigen/Core>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <vector>

namespace Acts {
// using Vector3DMap = Eigen::Map<Vector3D>;

/// @brief Simple class holding result of propagation call
///
/// @tparam parameters_t Type of final track parameters
/// @tparam result_t  Result for additional propagation
///                      quantity
struct PropagatorResult {
  PropagatorResult() = default;

  // The direct navigator initializer result
  DirectNavigatorInitializer::result_type initializerResult;

  // std::unique_ptr<const parameters_t> endParameters = nullptr;

  BoundMatrix transportJacobian{BoundMatrix::Zero()};

  /// Number of propagation steps that were carried out
  unsigned int steps = 0;

  /// Signed distance over which the parameters were propagated
  double pathLength = 0.;
};

/// @brief Options for propagate() call
///
template <typename action_t, typename aborter_t> struct PropagatorOptions {
  using action_type = action_t;

  /// Default constructor
  PropagatorOptions() = default;

  /// PropagatorOptions copy constructor
  PropagatorOptions(const PropagatorOptions<action_t, aborter_t> &po) = default;

  /// PropagatorOptions with context
  ACTS_DEVICE_FUNC PropagatorOptions(const GeometryContext &gctx,
                                     const MagneticFieldContext &mctx)
      : geoContext(gctx), magFieldContext(mctx) {}

  /// Propagation direction
  NavigationDirection direction = forward;

  /// The |pdg| code for (eventual) material integration - muon default
  int absPdgCode = 13;

  /// The mass for the particle for (eventual) material integration
  double mass = 105.6583755 * Acts::UnitConstants::MeV;

  /// Maximum number of steps for one propagate call
  unsigned int maxSteps = 1000;

  /// Maximum number of Runge-Kutta steps for the stepper step call
  unsigned int maxRungeKuttaStepTrials = 10000;

  /// Absolute maximum step size
  double maxStepSize = std::numeric_limits<double>::max();

  /// Absolute maximum path length
  double pathLimit = std::numeric_limits<double>::max();

  /// Required tolerance to reach target (surface, pathlength)
  double targetTolerance = s_onSurfaceTolerance;

  // Configurations for Stepper
  /// Tolerance for the error of the integration
  double tolerance = 1e-4;

  /// Cut-off value for the step size
  double stepSizeCutOff = 0.;

  /// The single actor
  action_t action;

  /// The single aborter
  aborter_t aborter;

  /// The navigator initializer
  DirectNavigatorInitializer initializer;

  /// The context object for the geometry
  GeometryContext geoContext;

  /// The context object for the magnetic field
  MagneticFieldContext magFieldContext;
};

/// @brief Propagator for particles (optionally in a magnetic field)
///
template <typename stepper_t,
          typename navigator_t = DirectNavigator<PlaneSurface<InfiniteBounds>>>
class Propagator final {
public:
  using Jacobian = BoundMatrix;

  /// Type of state object used by the propagation implementation
  using StepperState = typename stepper_t::State;

  /// Typedef the navigator state
  using NavigatorState = typename navigator_t::State;

  /// @brief private Propagator state for navigation and debugging
  ///
  /// @tparam propagator_options_t Type of the Objections object
  ///
  /// This struct holds the common state information for propagating
  /// which is independent of the actual stepper implementation.
  template <typename propagator_options_t> struct State {

    /// The default constructor
    State() = default;

    /// Create the propagator state from the options
    ///
    /// @tparam parameters_t the type of the start parameters
    /// @tparam propagator_options_t the type of the propagator options
    ///
    /// @param start The start parameters, used to initialize stepping state
    /// @param topts The options handed over by the propagate call
    template <typename parameters_t>
    ACTS_DEVICE_FUNC State(const parameters_t &start,
                           const propagator_options_t &topts)
        : options(topts), stepping(topts.geoContext, start, topts.direction,
                                   topts.maxStepSize, topts.tolerance),
          geoContext(topts.geoContext) {
      // Setting the start surface
      navigation.startSurface = &start.referenceSurface();
    }

    /// These are the options - provided for each propagation step
    propagator_options_t options;

    /// Stepper state - internal state of the Stepper
    StepperState stepping;

    /// Navigation state - internal state of the Navigator
    NavigatorState navigation;

    /// The context object for the geometry
    GeometryContext geoContext;
  };

  /// Constructor from implementation object
  ///
  /// @param stepper The stepper implementation is moved to a private member
  ACTS_DEVICE_FUNC explicit Propagator(stepper_t stepper,
                                       navigator_t navigator = navigator_t())
      : m_stepper(std::move(stepper)), m_navigator(std::move(navigator)) {}

  /// @brief Propagate track parameters
  ///
  template <typename parameters_t, typename propagator_options_t,
            typename path_aborter_t = PathLimitReached>
  ACTS_DEVICE_FUNC PropagatorResult propagate(
      const parameters_t &start, const propagator_options_t &options,
      typename propagator_options_t::action_type::result_type &actorResult)
      const;

#ifdef __CUDACC__
  /// @brief Propagate track parameters (device only function)
  ///
  template <typename parameters_t, typename propagator_options_t,
            typename path_aborter_t = PathLimitReached>
  __device__ void propagate(
      const parameters_t &start, const propagator_options_t &options,
      typename propagator_options_t::action_type::result_type &actorResult,
      PropagatorResult &result) const;
#endif

  /// @brief Get a non-const reference on the underlying stepper
  ///
  /// @return stepper reference
  ACTS_DEVICE_FUNC stepper_t &refStepper() { return m_stepper; }

private:
  /// Implementation of propagation algorithm
  stepper_t m_stepper;

  /// Implementation of navigator
  navigator_t m_navigator;
};
} // namespace Acts

#include "Propagator.ipp"
