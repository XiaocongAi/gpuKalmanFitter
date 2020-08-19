#include "Propagator/detail/CovarianceEngine.hpp"
#include "Utilities/ParameterDefinitions.hpp"

namespace Acts {
namespace detail {

/// @brief Storage of magnetic field and the sub steps during a RKN4 step
struct StepData {
  /// Magnetic field evaulations
  Vector3D B_first, B_middle, B_last;
  /// k_i of the RKN4 algorithm
  Vector3D k1, k2, k3, k4;
};

// place functions in order of invocation

// 1) The following functor evaluates k_i of RKN4
template <typename propagator_state_t>
ACTS_DEVICE_FUNC auto evaluatek(const propagator_state_t &state,
                                const Vector3D &bField, const int i = 0,
                                const double h = 0.,
                                const Vector3D &kprev = Vector3D(0, 0, 0))
    -> Vector3D {
  Vector3D knew;
  auto qop = state.stepping.q / state.stepping.p;
  // First step does not rely on previous data
  if (i == 0) {
    knew = qop * (state.stepping.dir).cross(bField);
  } else {
    knew = qop * ((state.stepping.dir) + h * kprev).cross(bField);
  }
  return knew;
}

// 2) The propagation function for time coordinate
template <typename propagator_state_t>
ACTS_DEVICE_FUNC void propagationTime(propagator_state_t &state,
                                      const double &h) {
  /// This evaluation is based on dt/ds = 1/v = 1/(beta * c) with the velocity
  /// v, the speed of light c and beta = v/c. This can be re-written as dt/ds
  /// = sqrt(m^2/p^2 + c^{-2}) with the mass m and the momentum p.
  auto derivative = std::hypot(1, state.options.mass / state.stepping.p);
  state.stepping.t += h * derivative;
  if (state.stepping.covTransport) {
    state.stepping.derivative(3) = derivative;
  }
}

// 3) The following functor calculates the transport matrix D for the jacobian
template <typename propagator_state_t>
ACTS_DEVICE_FUNC void transportMatrix(const propagator_state_t &state,
                                      const StepData &sd, const double &h,
				      FreeMatrix& D)
{
  D = FreeMatrix::Identity();
  auto dir = state.stepping.dir;
  auto qop = state.stepping.q / state.stepping.p;

  const double half_h = h * 0.5;
  // This sets the reference to the sub matrices
  // dFdx is already initialised as (3x3) idendity

  // For the case without energy loss
  ActsVectorD<3> dk1dL = dir.cross(sd.B_first);
  ActsVectorD<3> dk2dL = (dir + half_h * sd.k1).cross(sd.B_middle) +
          qop * half_h * dk1dL.cross(sd.B_middle);
  ActsVectorD<3> dk3dL = (dir + half_h * sd.k2).cross(sd.B_middle) +
          qop * half_h * dk2dL.cross(sd.B_middle);
  ActsVectorD<3> dk4dL = (dir + h * sd.k3).cross(sd.B_last) + qop * h * dk3dL.cross(sd.B_last);

  ActsMatrixD<3, 3> dk1dT = ActsMatrixD<3, 3>::Zero();
{
  dk1dT(0, 1) = sd.B_first.z();
  dk1dT(0, 2) = -sd.B_first.y();
  dk1dT(1, 0) = -sd.B_first.z();
  dk1dT(1, 2) = sd.B_first.x();
  dk1dT(2, 0) = sd.B_first.y();
  dk1dT(2, 1) = -sd.B_first.x();
  dk1dT *= qop;
}

  ActsMatrixD<3, 3> dk2dT = ActsMatrixD<3, 3>::Identity();
{
  dk2dT += half_h * dk1dT;
  dk2dT = qop * VectorHelpers::cross(dk2dT, sd.B_middle);
}
  ActsMatrixD<3, 3> dk3dT = ActsMatrixD<3, 3>::Identity();
{
  dk3dT += half_h * dk2dT;
  dk3dT = qop * VectorHelpers::cross(dk3dT, sd.B_middle);
}
  ActsMatrixD<3, 3> dk4dT = ActsMatrixD<3, 3>::Identity();
{
  dk4dT += h * dk3dT;
  dk4dT = qop * VectorHelpers::cross(dk4dT, sd.B_last);
}
{
  auto dFdT = D.block<3, 3>(0, 4);
  dFdT.setIdentity();
  dFdT += h / 6. * (dk1dT + dk2dT + dk3dT);
  dFdT *= h;
}
{
  auto dFdL = D.block<3, 1>(0, 7);
  dFdL = (h * h) / 6. * (dk1dL + dk2dL + dk3dL);
}
{
  // dGdx is already initialised as (3x3) zero
  auto dGdT = D.block<3, 3>(4, 4);
  dGdT += h / 6. * (dk1dT + 2. * (dk2dT + dk3dT) + dk4dT);
}
{
  auto dGdL = D.block<3, 1>(4, 7);
  dGdL = h / 6. * (dk1dL + 2. * (dk2dL + dk3dL) + dk4dL);
}
  D(3, 7) = h * state.options.mass * state.options.mass * state.stepping.q /
            (state.stepping.p *
             std::hypot(1., state.options.mass / state.stepping.p));
}

} // namespace detail
} // namespace Acts

template <typename B>
template <typename propagator_state_t>
ACTS_DEVICE_FUNC bool
Acts::EigenStepper<B>::step(propagator_state_t &state) const {

  const bool IS_MAIN_THREAD = threadIdx.x == 0 && threadIdx.y == 0;

  // Construt a stepping data here
  __shared__ detail::StepData sd;
  // Default constructor will result in wrong value on GPU
  double error_estimate = 0.;

  // First Runge-Kutta point (at current position)
  if (IS_MAIN_THREAD) {
    sd.B_first = getField(state.stepping, state.stepping.pos);
    sd.k1 = detail::evaluatek(state, sd.B_first, 0);
  }
  __syncthreads();

  // The following functor starts to perform a Runge-Kutta step of a certain
  // size, going up to the point where it can return an estimate of the local
  // integration error. The results are stated in the local variables above,
  // allowing integration to continue once the error is deemed satisfactory
  const auto tryRungeKuttaStep = [&](const ConstrainedStep &h) -> bool {
    // State the square and half of the step size
    const double h2 = h * h;
    const double half_h = h * 0.5;

    // Second Runge-Kutta point
    const Vector3D pos1 =
        state.stepping.pos + half_h * state.stepping.dir + h2 * 0.125 * sd.k1;
    sd.B_middle = getField(state.stepping, pos1);
    sd.k2 = detail::evaluatek(state, sd.B_middle, 1, half_h, sd.k1);

    // Third Runge-Kutta point
    sd.k3 = detail::evaluatek(state, sd.B_middle, 2, half_h, sd.k2);

    // Last Runge-Kutta point
    const Vector3D pos2 =
        state.stepping.pos + h * state.stepping.dir + h2 * 0.5 * sd.k3;
    sd.B_last = getField(state.stepping, pos2);
    sd.k4 = detail::evaluatek(state, sd.B_last, 3, h, sd.k3);

    // Compute and check the local integration error estimate
    // @Todo
    error_estimate = std::max(
        h2 * (sd.k1 - sd.k2 - sd.k3 + sd.k4).template lpNorm<1>(), 1e-20);
    return (error_estimate <= state.options.tolerance);
  };


  __shared__ bool earlyExit;

  if (IS_MAIN_THREAD) {
    double stepSizeScaling = 1.;
    size_t nStepTrials = 0;

    earlyExit = false;
    // Select and adjust the appropriate Runge-Kutta step size as given
    // ATL-SOFT-PUB-2009-001
    while (!tryRungeKuttaStep(state.stepping.stepSize)) {
      stepSizeScaling =
        std::min(std::max(0.25, std::pow((state.options.tolerance /
                                          std::abs(2. * error_estimate)),
                                         0.25)),
                 4.);
      // if (stepSizeScaling == 1.) {
      // break;
      //}
      state.stepping.stepSize = state.stepping.stepSize * stepSizeScaling;
      
      // Todo: adapted error handling on GPU?
      // If step size becomes too small the particle remains at the initial
      // place
      if (state.stepping.stepSize * state.stepping.stepSize <
	  state.options.stepSizeCutOff * state.options.stepSizeCutOff) {
	// Not moving due to too low momentum needs an aborter
	earlyExit = true;
	break;
      }
      
      // If the parameter is off track too much or given stepSize is not
      // appropriate
      if (nStepTrials > state.options.maxRungeKuttaStepTrials) {
	// Too many trials, have to abort
	earlyExit = true;
	break;
      }
      nStepTrials++;
    }
  }
  __syncthreads();
  
  if (earlyExit) {
    return false;
  }
  
  // use the adjusted step size
  const double h = state.stepping.stepSize;

  if (IS_MAIN_THREAD) {
    // Propagate the time
    detail::propagationTime(state, h);
  }
  __syncthreads();
  
  __shared__ FreeMatrix jacTransport;
  // When doing error propagation, update the associated Jacobian matrix
  // The step transport matrix in global coordinates
  if (state.stepping.covTransport) {
    jacTransport(threadIdx.x, threadIdx.y) = state.stepping.jacTransport(threadIdx.x, threadIdx.y);

    // for moment, only update the transport part
    detail::transportMatrix(state, sd, h, jacTransport);
    
    double acc = 0.0;
    for (int i = 0; i < 8; ++i) {
      acc += jacTransport(threadIdx.x, i) * state.stepping.jacTransport(i, threadIdx.y);
    }
    state.stepping.jacTransport(threadIdx.x, threadIdx.y) = acc;
  }

  if (IS_MAIN_THREAD) {
    // Update the track parameters according to the equations of motion
    state.stepping.pos +=
      h * state.stepping.dir + h * h / 6. * (sd.k1 + sd.k2 + sd.k3);
    state.stepping.dir += h / 6. * (sd.k1 + 2. * (sd.k2 + sd.k3) + sd.k4);
    state.stepping.dir /= state.stepping.dir.norm();
    if (state.stepping.covTransport) {
      state.stepping.derivative.template head<3>() = state.stepping.dir;
      state.stepping.derivative.template segment<3>(4) = sd.k4;
    }
    state.stepping.pathAccumulated += h;
  }
  __syncthreads();
  // return h;
  return true;
}

template <typename B>
ACTS_DEVICE_FUNC auto
Acts::EigenStepper<B>::boundState(State &state, const Surface &surface) const
    -> BoundState {
  FreeVector parameters;
  parameters[0] = state.pos[0];
  parameters[1] = state.pos[1];
  parameters[2] = state.pos[2];
  parameters[3] = state.t;
  parameters[4] = state.dir[0];
  parameters[5] = state.dir[1];
  parameters[6] = state.dir[2];
  parameters[7] = state.q / state.p;

  return detail::boundState(state.geoContext, state.cov, state.jacobian,
                            state.jacTransport, state.derivative,
                            state.jacToGlobal, parameters, state.covTransport,
                            state.pathAccumulated, surface);
}

template <typename B>
ACTS_DEVICE_FUNC auto
Acts::EigenStepper<B>::curvilinearState(State &state) const
    -> CurvilinearState {
  FreeVector parameters;
  parameters << state.pos[0], state.pos[1], state.pos[2], state.t, state.dir[0],
      state.dir[1], state.dir[2], state.q / state.p;
  return detail::curvilinearState(
      state.cov, state.jacobian, state.jacTransport, state.derivative,
      state.jacToGlobal, parameters, state.covTransport, state.pathAccumulated);
}

template <typename B>
ACTS_DEVICE_FUNC void
Acts::EigenStepper<B>::update(State &state, const FreeVector &parameters,
                              const Covariance &covariance) const {
  state.pos = parameters.template segment<3>(eFreePos0);
  state.dir = parameters.template segment<3>(eFreeDir0).normalized();
  state.p = std::abs(1. / parameters[eFreeQOverP]);
  state.t = parameters[eFreeTime];

  state.cov = covariance;
}

template <typename B>
ACTS_DEVICE_FUNC void
Acts::EigenStepper<B>::update(State &state, const Vector3D &uposition,
                              const Vector3D &udirection, double up,
                              double time) const {
  state.pos = uposition;
  state.dir = udirection;
  state.p = up;
  state.t = time;
}

template <typename B>
ACTS_DEVICE_FUNC void
Acts::EigenStepper<B>::covarianceTransport(State &state) const {
  detail::covarianceTransport(state.cov, state.jacobian, state.jacTransport,
                              state.derivative, state.jacToGlobal, state.dir);
}

template <typename B>
ACTS_DEVICE_FUNC void
Acts::EigenStepper<B>::covarianceTransport(State &state,
                                           const Surface &surface) const {
  FreeVector parameters;
  parameters[0] = state.pos[0];
  parameters[1] = state.pos[1];
  parameters[2] = state.pos[2];
  parameters[3] = state.t;
  parameters[4] = state.dir[0];
  parameters[5] = state.dir[1];
  parameters[6] = state.dir[2];
  parameters[7] = state.q / state.p;
  detail::covarianceTransport(state.geoContext, state.cov, state.jacobian,
                              state.jacTransport, state.derivative,
                              state.jacToGlobal, parameters, surface);
}
