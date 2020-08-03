#pragma once

#include "Utilities/Definitions.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Helpers.hpp"
#include <iostream>
#include <numeric>

namespace Acts {
template <typename bfield_t> struct EigenStepper {
  /// Jacobian and Covariance defintions
  using Jacobian = BoundMatrix;
  using BField = bfield_t;
  using Covariance = BoundSymMatrix;

  /// @brief State for track parameter propagation
  ///
  /// It contains the stepping information and is provided thread local
  /// by the propagator
  struct State {
    /// Constructor
    template <typename parameters_t>
    ACTS_DEVICE_FUNC explicit State(
        const parameters_t &par, NavigationDirection ndir = forward,
        double ssize = std::numeric_limits<double>::max(),
        double stolerance = 0.01)
        : pos(par.position()), dir(par.momentum().normalized()),
          q(par.charge()), p(par.momentum().norm()), navDir(ndir),
          stepSize(ndir * std::abs(ssize)), tolerance(stolerance) {}

    /// Global particle position
    Vector3D pos = Vector3D(0., 0., 0.);

    /// Momentum direction (normalized)
    Vector3D dir = Vector3D(1., 0., 0.);

    /// Momentum
    double p = 0.;

    /// The charge
    int q = 1;

    /// Propagated time
    double t = 0.;

    /// Navigation direction, this is needed for searching
    NavigationDirection navDir;

    /// The full jacobian of the transport entire transport
    Jacobian jacobian = Jacobian::Identity();

    /// Jacobian from local to the global frame
    BoundToFreeMatrix jacToGlobal = BoundToFreeMatrix::Zero();

    /// Pure transport jacobian part from runge kutta integration
    FreeMatrix jacTransport = FreeMatrix::Identity();

    /// The propagation derivative
    FreeVector derivative = FreeVector::Zero();

    /// Covariance matrix (and indicator)
    //// associated with the initial error on track parameters
    bool covTransport = false;
    Covariance cov = Covariance::Zero();

    /// Accummulated path length state
    double pathAccumulated = 0.;

    /// The tolerance for the stepping
    double tolerance = 0.01;

    /// Adaptive step size of the runge-kutta integration
    double stepSize = 1000;

    /// @brief Storage of magnetic field and the sub steps during a RKN4 step
    struct {
      /// Magnetic field evaulations
      Vector3D B_first, B_middle, B_last;
      /// k_i of the RKN4 algorithm
      Vector3D k1, k2, k3, k4;
    } stepData;
  };

  /// Constructor requires knowledge of the detector's magnetic field
  ACTS_DEVICE_FUNC EigenStepper(BField bField = BField())
      : m_bField(std::move(bField)) {}

  /// Get the field for the stepping, it checks first if the access is still
  /// within the Cell, and updates the cell if necessary.
  ///
  /// @param [in,out] state is the propagation state associated with the track
  ///                 the magnetic field cell is used (and potentially updated)
  /// @param [in] pos is the field position
  ACTS_DEVICE_FUNC Vector3D getField(State & /*state*/,
                                     const Vector3D &pos) const {
    // get the field from the cell
    return m_bField.getField(pos);
  }

  /// @brief Get a non-const reference on the underlying bField
  ///
  /// @return bField reference
  ACTS_DEVICE_FUNC BField &refField() { return m_bField; }

  /// Perform a Runge-Kutta track parameter propagation step
  ///
  /// @param [in,out] state is the propagation state associated with the track
  /// parameters that are being propagated.
  ///
  ///                      the state contains the desired step size.
  ///                      It can be negative during backwards track
  ///                      propagation,
  ///                      and since we're using an adaptive algorithm, it can
  ///                      be modified by the stepper class during propagation.
  template <typename propagator_state_t>
  ACTS_DEVICE_FUNC double step(propagator_state_t &state) const;

private:
  /// Magnetic field inside of the detector
  BField m_bField;

  /// Overstep limit: could/should be dynamic
  double m_overstepLimit = 0.01;
};
} // namespace Acts

#include "Propagator/EigenStepper.ipp"
