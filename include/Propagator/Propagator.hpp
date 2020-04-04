#pragma once

#include "Utilities/Definitions.hpp"
#include <Eigen/Core>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <vector>

namespace Acts {
using Vector3DMap = Eigen::Map<Vector3D>;

/// @brief Propagator result
///
template <unsigned int nSteps> struct PropagatorResult {
  PropagatorResult() = default;
  Eigen::Array<double, 3, nSteps> position;
  Eigen::Array<double, 3, nSteps> momentum;
  ACTS_DEVICE_FUNC int steps() { return nSteps; }
};

/// @brief Options for propagate() call
///
struct PropagatorOptions {

  /// Default constructor
  PropagatorOptions() = default;

  /// Propagation direction
  NavigationDirection direction = forward;

  /// The |pdg| code for (eventual) material integration - pion default
  int absPdgCode = 211;

  /// The mass for the particle for (eventual) material integration
  double mass = 139.57018;

  /// Maximum number of steps for one propagate call
  unsigned int maxSteps = 1000;

  /// Maximum number of Runge-Kutta steps for the stepper step call
  unsigned int maxRungeKuttaStepTrials = 10000;

  /// Absolute maximum step size
  double maxStepSize = std::numeric_limits<double>::max();

  /// Absolute maximum path length
  double pathLimit = std::numeric_limits<double>::max();

  // Configurations for Stepper
  /// Tolerance for the error of the integration
  double tolerance = 1e-4;

  /// Cut-off value for the step size
  double stepSizeCutOff = 0.;
};

/// @brief Propagator for particles (optionally in a magnetic field)
///
template <typename stepper_t> class Propagator final {
public:
  /// Type of state object used by the propagation implementation
  using StepperState = typename stepper_t::State;

  /// @brief private Propagator state for navigation and debugging
  ///
  /// @tparam propagator_options_t Type of the Objections object
  ///
  /// This struct holds the common state information for propagating
  /// which is independent of the actual stepper implementation.
  template <typename propagator_options_t> struct State {
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
        : options(topts),
          stepping(start, topts.direction, topts.maxStepSize, topts.tolerance) {
    }

    /// These are the options - provided for each propagation step
    propagator_options_t options;

    /// Stepper state - internal state of the Stepper
    StepperState stepping;
  };

  /// Constructor from implementation object
  ///
  /// @param stepper The stepper implementation is moved to a private member
  ACTS_DEVICE_FUNC explicit Propagator(stepper_t stepper)
      : m_stepper(std::move(stepper)) {}

  /// @brief Propagate track parameters
  ///
  template <typename parameters_t, typename propagator_options_t,
            typename result_t>
  ACTS_DEVICE_FUNC void propagate(const parameters_t &start,
                                  const propagator_options_t &options,
                                  result_t &result) const;

private:
  /// Implementation of propagation algorithm
  stepper_t m_stepper;
};
} // namespace Acts

#include "Propagator.ipp"
