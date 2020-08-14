#pragma once

#include "EventData/TrackParameters.hpp"
#include "Propagator/ConstrainedStep.hpp"
#include "Propagator/detail/SteppingHelper.hpp"
#include "Surfaces/Surface.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Helpers.hpp"
#include "Utilities/Intersection.hpp"
#include "Utilities/ParameterDefinitions.hpp"

#include <iostream>
#include <numeric>

namespace Acts {
template <typename bfield_t> struct EigenStepper {
  /// Jacobian and Covariance defintions
  using Jacobian = BoundMatrix;
  using Covariance = BoundSymMatrix;
  using BoundState = std::tuple<BoundParameters, Jacobian, double>;
  using CurvilinearState = std::tuple<CurvilinearParameters, Jacobian, double>;
  using BField = bfield_t;

  /// @brief State for track parameter propagation
  ///
  /// It contains the stepping information and is provided thread local
  /// by the propagator
  struct State {
    /// Default constructor - deleted
    State() = delete;

    /// Constructor from the initial track parameters
    ///
    /// @param [in] gctx is the context object for the geometry
    /// @param [in] par The track parameters at start
    /// @param [in] ndir The navigation direciton w.r.t momentum
    /// @param [in] ssize is the maximum step size
    /// @param [in] stolerance is the stepping tolerance
    ///
    /// @note the covariance matrix is copied when needed
    template <typename parameters_t>
    ACTS_DEVICE_FUNC explicit State(
        const GeometryContext &gctx, const parameters_t &par,
        NavigationDirection ndir = forward,
        double ssize = std::numeric_limits<double>::max(),
        double stolerance = s_onSurfaceTolerance)
        : pos(par.position()), dir(par.momentum().normalized()),
          p(par.momentum().norm()), q(par.charge()), t(par.time()),
          navDir(ndir), stepSize(ndir * std::abs(ssize)), tolerance(stolerance),
          geoContext(gctx) {
      // Init the jacobian matrix if needed
      if (par.covariance()) {
        // Get the reference surface for navigation
        const auto &surface = par.referenceSurface();
        // set the covariance transport flag to true and copy
        covTransport = true;
        cov = BoundSymMatrix(*par.covariance());
        surface.initJacobianToGlobal(gctx, jacToGlobal, pos, dir,
                                     par.parameters());
      }
    }

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

    /// Adaptive step size of the runge-kutta integration
    ConstrainedStep stepSize{std::numeric_limits<double>::max()};

    /// Last performed step (for overstep limit calculation)
    double previousStepSize = 0.;

    /// The tolerance for the stepping
    double tolerance = s_onSurfaceTolerance;

    /// The geometry context
    const GeometryContext &geoContext;

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
  ACTS_DEVICE_FUNC bool step(propagator_state_t &state) const;

  /// Global particle position accessor
  ///
  /// @param state [in] The stepping state (thread-local cache)
  ACTS_DEVICE_FUNC Vector3D position(const State &state) const {
    return state.pos;
  }

  /// Momentum direction accessor
  ///
  /// @param state [in] The stepping state (thread-local cache)
  ACTS_DEVICE_FUNC Vector3D direction(const State &state) const {
    return state.dir;
  }

  /// Actual momentum accessor
  ///
  /// @param state [in] The stepping state (thread-local cache)
  ACTS_DEVICE_FUNC double momentum(const State &state) const { return state.p; }

  /// Charge access
  ///
  /// @param state [in] The stepping state (thread-local cache)
  ACTS_DEVICE_FUNC double charge(const State &state) const { return state.q; }

  /// Time access
  ///
  /// @param state [in] The stepping state (thread-local cache)
  ACTS_DEVICE_FUNC double time(const State &state) const { return state.t; }

  /// Update surface status
  ///
  /// It checks the status to the reference surface & updates
  /// the step size accordingly
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  /// @param surface [in] The surface provided
  /// @param bcheck [in] The boundary check for this status update
  ACTS_DEVICE_FUNC Intersection::Status
  updateSurfaceStatus(State &state, const Surface &surface,
                      const BoundaryCheck &bcheck) const {
    return detail::updateSingleSurfaceStatus<EigenStepper>(*this, state,
                                                           surface, bcheck);
  }

  /// Update step size
  ///
  /// This method intersects the provided surface and update the navigation
  /// step estimation accordingly (hence it changes the state). It also
  /// returns the status of the intersection to trigger onSurface in case
  /// the surface is reached.
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  /// @param oIntersection [in] The ObjectIntersection to layer, boundary, etc
  /// @param release [in] boolean to trigger step size release
  template <typename object_intersection_t>
  ACTS_DEVICE_FUNC void
  updateStepSize(State &state, const object_intersection_t &oIntersection,
                 bool release = true) const {
    detail::updateSingleStepSize<EigenStepper>(state, oIntersection, release);
  }

  /// Set Step size - explicitely with a double
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  /// @param stepSize [in] The step size value
  /// @param stype [in] The step size type to be set
  ACTS_DEVICE_FUNC void
  setStepSize(State &state, double stepSize,
              ConstrainedStep::Type stype = ConstrainedStep::actor) const {
    state.previousStepSize = state.stepSize;
    state.stepSize.update(stepSize, stype, true);
  }

  /// Release the Step size
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  ACTS_DEVICE_FUNC void releaseStepSize(State &state) const {
    state.stepSize.release(ConstrainedStep::actor);
  }

  /// Output the Step Size - single component
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  std::string outputStepSize(const State &state) const {
    return state.stepSize.toString();
  }

  /// Overstep limit
  ///
  /// @param state [in] The stepping state (thread-local cache)
  ACTS_DEVICE_FUNC double overstepLimit(const State & /*state*/) const {
    // A dynamic overstep limit could sit here
    return -m_overstepLimit;
  }

  /// Create and return the bound state at the current position
  ///
  /// @brief This transports (if necessary) the covariance
  /// to the surface and creates a bound state. It does not check
  /// if the transported state is at the surface, this needs to
  /// be guaranteed by the propagator
  ///
  /// @param [in] state State that will be presented as @c BoundState
  /// @param [in] surface The surface to which we bind the state
  ///
  /// @return A bound state:
  ///   - the parameters at the surface
  ///   - the stepwise jacobian towards it (from last bound)
  ///   - and the path length (from start - for ordering)
  ACTS_DEVICE_FUNC BoundState boundState(State &state,
                                         const Surface &surface) const;

  /// Create and return a curvilinear state at the current position
  ///
  /// @brief This transports (if necessary) the covariance
  /// to the current position and creates a curvilinear state.
  ///
  /// @param [in] state State that will be presented as @c CurvilinearState
  ///
  /// @return A curvilinear state:
  ///   - the curvilinear parameters at given position
  ///   - the stepweise jacobian towards it (from last bound)
  ///   - and the path length (from start - for ordering)
  ACTS_DEVICE_FUNC CurvilinearState curvilinearState(State &state) const;

  /// Method to update a stepper state to the some parameters
  ///
  /// @param [in,out] state State object that will be updated
  /// @param [in] pars Parameters that will be written into @p state
  ACTS_DEVICE_FUNC void update(State &state, const FreeVector &parameters,
                               const Covariance &covariance) const;

  /// Method to update momentum, direction and p
  ///
  /// @param [in,out] state State object that will be updated
  /// @param [in] uposition the updated position
  /// @param [in] udirection the updated direction
  /// @param [in] up the updated momentum value
  ACTS_DEVICE_FUNC void update(State &state, const Vector3D &uposition,
                               const Vector3D &udirection, double up,
                               double time) const;

  /// Method for on-demand transport of the covariance
  /// to a new curvilinear frame at current  position,
  /// or direction of the state
  ///
  /// @param [in,out] state State of the stepper
  ///
  /// @return the full transport jacobian
  ACTS_DEVICE_FUNC void covarianceTransport(State &state) const;

  /// Method for on-demand transport of the covariance
  /// to a new curvilinear frame at current position,
  /// or direction of the state
  ///
  /// @tparam surface_t the Surface type
  ///
  /// @param [in,out] state State of the stepper
  /// @param [in] surface is the surface to which the covariance is forwarded to
  /// @note no check is done if the position is actually on the surface
  ACTS_DEVICE_FUNC void covarianceTransport(State &state,
                                            const Surface &surface) const;

private:
  /// Magnetic field inside of the detector
  BField m_bField;

  /// Overstep limit: could/should be dynamic
  double m_overstepLimit = 0.01;
};
} // namespace Acts

#include "Propagator/EigenStepper.ipp"
