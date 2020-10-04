// This file is part of the Acts project.
//
// Copyright (C) 2016-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EventData/TrackParameters.hpp"
#include "EventData/TrackState.hpp"
#include "Fitter/detail/VoidKalmanComponents.hpp"
#include "Geometry/GeometryContext.hpp"
#include "MagneticField/MagneticFieldContext.hpp"
#include "Propagator/ConstrainedStep.hpp"
#include "Propagator/DirectNavigator.hpp"
#include "Propagator/Propagator.hpp"
#include "Propagator/StandardAborters.hpp"
#include "Propagator/detail/CovarianceEngine.hpp"
#include "Utilities/CudaKernelContainer.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Profiling.hpp"

#include <functional>
#include <memory>

namespace Acts {

/// @brief A light-weight surface finder
struct SurfaceFinder {
  const Surface *surface = nullptr;

  template <typename source_link_t>
  ACTS_DEVICE_FUNC bool operator()(const source_link_t &sl) {
    if (surface) {
      return sl.referenceSurface() == *surface;
    }
    return false;
  }
};

/// @brief Options struct how the Fitter is called
///
/// It contains the context of the fitter call, the outlier finder, the
/// optional surface where to express the fit result and configurations for
/// material effects and smoothing options
///
///
/// @note the context objects must be provided
template <typename outlier_finder_t = VoidOutlierFinder>
struct KalmanFitterOptions {
  // Broadcast the outlier finder type
  using OutlierFinder = outlier_finder_t;

  /// Deleted default constructor
  // KalmanFitterOptions() = delete;
  KalmanFitterOptions() = default;

  /// PropagatorOptions with context
  ///
  /// @param gctx The goemetry context for this fit
  /// @param mctx The magnetic context for this fit
  /// @param cctx The calibration context for this fit
  /// @param olCfg The config for the outlier finder
  /// @param rSurface The reference surface for the fit to be expressed at
  /// @param mScattering Whether to include multiple scattering
  /// @param eLoss Whether to include energy loss
  /// @param bwdFiltering Whether to run backward filtering as smoothing
  ACTS_DEVICE_FUNC
  KalmanFitterOptions(const GeometryContext &gctx,
                      const MagneticFieldContext &mctx,
                      const OutlierFinder &outlierFinder_ = VoidOutlierFinder(),
                      Surface *rSurface = nullptr)
      : geoContext(gctx), magFieldContext(mctx), outlierFinder(outlierFinder_),
        referenceSurface(rSurface) {}

  /// Context object for the geometry
  GeometryContext geoContext;
  /// Context object for the magnetic field
  MagneticFieldContext magFieldContext;

  /// The config for the outlier finder
  OutlierFinder outlierFinder;

  /// The reference Surface
  Surface *referenceSurface = nullptr;
};

template <typename source_link_t, typename parameters_t>
struct KalmanFitterResult {
  using TrackStateType = TrackState<source_link_t, parameters_t>;

  // Fitted states that the actor has handled.
  CudaKernelContainer<TrackStateType> fittedStates;

  // The optional Parameters at the provided surface
  // std::optional<BoundParameters> fittedParameters;

  // Counter for states with measurements
  size_t measurementStates = 0;

  // Indicator if smoothing has been done.
  bool smoothed = false;

  // Indicator if track fitting has been done
  bool finished = false;

  // Indicator if the fitting is successful
  bool result = true;
};

/// @brief Kalman fitter implementation of Acts as a plugin
///
/// to the Propgator
///
/// @tparam propagator_t Type of the propagation class
/// @tparam updater_t Type of the kalman updater class
/// @tparam smoother_t Type of the kalman smoother class
/// @tparam outlier_finder_t Type of the outlier finder class
/// @tparam calibrator_t Type of the calibrator class
///
/// The Kalman filter contains an Actor and a Sequencer sub-class.
/// The Sequencer has to be part of the Navigator of the Propagator
/// in order to initialize and provide the measurement surfaces.
///
/// The Actor is part of the Propagation call and does the Kalman update
/// and eventually the smoothing.  Updater, Smoother and Calibrator are
/// given to the Actor for further use:
/// - The Updater is the implemented kalman updater formalism, it
///   runs via a visitor pattern through the measurements.
/// - The Smoother is called at the end of the forward fit by the Actor.
/// - The outlier finder is called during the filtering by the Actor.
///   It determines if the measurement is an outlier
/// - The Calibrator is a dedicated calibration algorithm that allows
///   to calibrate measurements using track information, this could be
///    e.g. sagging for wires, module deformations, etc.
///
/// Measurements are not required to be ordered for the KalmanFilter,
/// measurement ordering needs to be figured out by the navigation of
/// the propagator.
///
/// The void components are provided mainly for unit testing.
template <typename propagator_t, typename updater_t = VoidKalmanUpdater,
          typename smoother_t = VoidKalmanSmoother,
          typename outlier_finder_t = VoidOutlierFinder>
class KalmanFitter {
public:
  /// Default constructor is deleted
  KalmanFitter() = delete;

  /// Constructor from arguments
  ACTS_DEVICE_FUNC KalmanFitter(propagator_t pPropagator)
      : m_propagator(std::move(pPropagator)) {}

private:
  /// The propgator for the transport and material update
  propagator_t m_propagator;

  /// @brief Propagator Actor plugin for the KalmanFilter
  ///
  /// @tparam source_link_t is an type fulfilling the @c SourceLinkConcept
  /// @tparam parameters_t The type of parameters used for "local" paremeters.
  ///
  /// The KalmanActor does not rely on the measurements to be
  /// sorted along the track.
  template <typename source_link_t, typename parameters_t> class Actor {
  public:
    /// Broadcast the result_type
    using result_type = KalmanFitterResult<source_link_t, parameters_t>;

    using TrackStateType = typename result_type::TrackStateType;

    /// Broadcast the input measurement container type
    using InputMeasurementsType = CudaKernelContainer<source_link_t>;
    // using InputMeasurementsType = std::vector<source_link_t>;

    /// The target surface
    const Surface *targetSurface = nullptr;

    /// Allows retrieving measurements for a surface
    InputMeasurementsType inputMeasurements;

    /// @brief Kalman actor operation
    ///
    /// @tparam propagator_state_t is the type of Propagagor state
    /// @tparam stepper_t Type of the stepper
    ///
    /// @param state is the mutable propagator state object
    /// @param stepper The stepper in use
    /// @param result is the mutable result state object
    template <typename propagator_state_t, typename stepper_t>
    ACTS_DEVICE_FUNC void operator()(propagator_state_t &state,
                                     const stepper_t &stepper,
                                     result_type &result) const {
      // printf("KalmanFitter step\n");

      // Update:
      // - Waiting for a current surface
      auto surface = state.navigation.currentSurface;
      if (surface != nullptr and !result.smoothed and !result.finished) {
        auto res = filter(surface, state, stepper, result);
        if (!res) {
          printf("Error in filter:\n");
          result.result = false;
        }
      }

      // Finalization:
      // when all track states have been handled or the navigation is breaked,
      // reset navigation&stepping before run backward filtering or
      // proceed to run smoothing
      if (result.measurementStates == inputMeasurements.size() or
          (result.measurementStates > 0 and state.navigation.navigationBreak)) {
        // printf("Finishing forward filtering");
        result.finished = true;
        // if (not result.smoothed) {
        //  printf("Finalize/run smoothing\n");
        //  auto res = finalize(state, stepper, result);
        //  if (!res) {
        //    printf("Error in finalize:\n");
        //    result.result = false;
        //  }
        //}
      }

      // Post-finalization:
      // - Progress to target/reference surface and built the final track
      // parameters
      if (result.smoothed) {
        // printf("Completing");
        // Remember the track fitting is done
        result.finished = true;
      }
    }

    /// @brief Kalman actor operation : update
    ///
    /// @tparam propagator_state_t is the type of Propagagor state
    /// @tparam stepper_t Type of the stepper
    ///
    /// @param surface The surface where the update happens
    /// @param state The mutable propagator state object
    /// @param stepper The stepper in use
    /// @param result The mutable result state object
    template <typename propagator_state_t, typename stepper_t>
    ACTS_DEVICE_FUNC bool
    filter(const Surface *surface, propagator_state_t &state,
           const stepper_t &stepper, result_type &result) const {
      // Try to find the surface in the measurement surfaces
      SurfaceFinder sFinder{surface};
      // auto sourcelink_it = std::find_if(inputMeasurements.begin(),
      // inputMeasurements.end(), sFinder);
      auto sourcelink_it = inputMeasurements.find_if(sFinder);
      if (sourcelink_it != inputMeasurements.end()) {
        // Screen output message
        auto center = (*surface).center(state.options.geoContext);
        // printf("Measurement surface detected with center = (%f, %f, %f)\n",
        // center.x(), center.y(), center.z());

        // create track state on the vector from sourcelink
        // result.fittedStates.push_back(TrackStateType(*sourcelink_it));
        // TrackStateType& trackState = result.fittedStates.back();
        result.fittedStates[result.measurementStates] =
            TrackStateType(*sourcelink_it);
        TrackStateType &trackState =
            result.fittedStates[result.measurementStates];
        auto pos = (*sourcelink_it).globalPosition(state.options.geoContext);
        // printf("sl position = (%f, %f, %f)\n", pos.x(), pos.y(), pos.z());

        // Transport & bind the state to the current surface
        // auto [boundParams, jacobian, pathLength] =
        auto bState = stepper.boundState(state.stepping, *surface);

        // Fill the track state
        trackState.parameter.predicted = std::move(bState.boundParams);
        trackState.parameter.jacobian = std::move(bState.jacobian);
        trackState.parameter.pathLength = std::move(bState.path);
        auto prePos = trackState.parameter.predicted.position();
        // printf("Predicted parameter position = (%f, %f, %f)\n", prePos.x(),
        // prePos.y(), prePos.z());

        // Get and set the type flags
        // auto& typeFlags = trackState.typeFlags();
        // typeFlags.set(TrackStateFlag::ParameterFlag);
        // typeFlags.set(TrackStateFlag::MeasurementFlag);

        // If the update is successful, set covariance and
        auto updateRes = m_updater(state.options.geoContext, trackState);
        if (!updateRes) {
          printf("Update step failed:\n");
          return false;
        }

        // Get the filtered parameters and update the stepping state
        const auto &filtered = trackState.parameter.filtered;
        // printf("Filtered parameter position = (%f, %f, %f)\n",
        // filtered.position().x(), filtered.position().y(),
        // filtered.position().z());
        stepper.update(state.stepping, filtered.position(),
                       filtered.momentum().normalized(),
                       filtered.momentum().norm(), filtered.time());

        // We count the state with measurement
        ++result.measurementStates;
      }
      return true;
    }

    /// @brief Kalman actor operation : finalize
    ///
    /// @tparam propagator_state_t is the type of Propagagor state
    /// @tparam stepper_t Type of the stepper
    ///
    /// @param state is the mutable propagator state object
    /// @param stepper The stepper in use
    /// @param result is the mutable result state object
    template <typename propagator_state_t, typename stepper_t>
    ACTS_DEVICE_FUNC bool finalize(propagator_state_t &state,
                                   const stepper_t &stepper,
                                   result_type &result) const {
      // Remember you smoothed the track states
      result.smoothed = true;

      // Smooth the track states
      const auto &smoothedPars =
          m_smoother(state.geoContext, result.fittedStates);

      if (smoothedPars) {
        stepper.update(state.stepping, *smoothedPars.position(),
                       *smoothedPars.momentum().normalized(),
                       *smoothedPars.momentum().norm(), *smoothedPars.time());

        // Reverse the propagation direction
        state.stepping.stepSize =
            ConstrainedStep(-1. * state.options.maxStepSize);
        state.stepping.navDir = backward;
        // Set accumulatd path to zero before targeting surface
        state.stepping.pathAccumulated = 0.;

        return true;
      }

      return false;
    }

    /// The Kalman updater
    updater_t m_updater;

    /// The Kalman smoother
    smoother_t m_smoother;

    /// The outlier finder
    outlier_finder_t m_outlierFinder;

    /// The Surface beeing
    SurfaceReached targetReached;
  };

  template <typename source_link_t, typename parameters_t> class Aborter {
  public:
    /// Broadcast the result_type
    using action_type = Actor<source_link_t, parameters_t>;

    template <typename propagator_state_t, typename stepper_t,
              typename result_t>
    ACTS_DEVICE_FUNC bool operator()(propagator_state_t & /*state*/,
                                     const stepper_t & /*stepper*/,
                                     const result_t &result) const {
      if (!result.result or result.finished) {
        return true;
      }
      return false;
    }
  };

public:
  /// Fit implementation of the foward filter, calls the
  /// the forward filter and backward smoother
  ///
  /// @tparam source_link_t Source link type identifying uncalibrated input
  /// measurements.
  /// @tparam start_parameters_t Type of the initial parameters
  /// @tparam parameters_t Type of parameters used for local parameters
  ///
  /// @param sourcelinks The fittable uncalibrated measurements
  /// @param sParameters The initial track parameters
  /// @param kfOptions KalmanOptions steering the fit
  /// @param kfResult The fitted result
  /// @param surfaceSequence The surface sequence to initialize the direct
  /// navigator
  /// @param surfaceSequenceSize The surface sequence size
  /// @note The input measurements are given in the form of @c SourceLinks.
  /// It's
  /// @c calibrator_t's job to turn them into calibrated measurements used in
  /// the fit.
  ///
  /// @return the output as an output track
  template <typename source_link_t, typename start_parameters_t,
            typename parameters_t = BoundParameters>
  ACTS_DEVICE_FUNC bool
  fit(const CudaKernelContainer<source_link_t> &sourcelinks,
      const start_parameters_t &sParameters,
      const KalmanFitterOptions<outlier_finder_t> &kfOptions,
      KalmanFitterResult<source_link_t, parameters_t> &kfResult,
      const Surface *surfaceSequence = nullptr,
      size_t surfaceSequenceSize = 0) const {

    PUSH_RANGE("fit", 0);

    // printf("Preparing %lu input measurements\n", sourcelinks.size());
    // Create the ActionList and AbortList
    using KalmanAborter = Aborter<source_link_t, parameters_t>;
    using KalmanActor = Actor<source_link_t, parameters_t>;

    // Create relevant options for the propagation options
    PropagatorOptions<KalmanActor, KalmanAborter> kalmanOptions(
        kfOptions.geoContext, kfOptions.magFieldContext);
    kalmanOptions.initializer.surfaceSequence = surfaceSequence;
    kalmanOptions.initializer.surfaceSequenceSize = surfaceSequenceSize;

    // Catch the actor and set the measurements
    auto &kalmanActor = kalmanOptions.action;
    kalmanActor.inputMeasurements = std::move(sourcelinks);
    kalmanActor.targetSurface = kfOptions.referenceSurface;

    // Set config for outlier finder
    kalmanActor.m_outlierFinder = kfOptions.outlierFinder;

    // Run the fitter
    const auto propRes =
        m_propagator.template propagate(sParameters, kalmanOptions, kfResult);

    POP_RANGE();

    if (!kfResult.result) {
      printf("KalmanFilter failed: \n");
      return false;
    }

    // Return the converted Track
    return true;
  }

#ifdef __CUDACC__
  /// Fit implementation of the foward filter, calls the
  /// the forward filter and backward smoother (device only version)
  ///
  /// @tparam source_link_t Source link type identifying uncalibrated input
  /// measurements.
  /// @tparam start_parameters_t Type of the initial parameters
  /// @tparam parameters_t Type of parameters used for local parameters
  ///
  /// @param sourcelinks The fittable uncalibrated measurements
  /// @param sParameters The initial track parameters
  /// @param kfOptions KalmanOptions steering the fit
  /// @param kfResult The fitted result
  /// @param surfaceSequence The surface sequence to initialize the direct
  /// navigator
  /// @param surfaceSequenceSize The surface sequence size
  /// @note The input measurements are given in the form of @c SourceLinks.
  /// It's
  /// @c calibrator_t's job to turn them into calibrated measurements used in
  /// the fit.
  ///
  /// @return the output as an output track
  template <typename source_link_t, typename start_parameters_t,
            typename parameters_t = BoundParameters>
  __device__ bool
  fitOnDevice(const CudaKernelContainer<source_link_t> &sourcelinks,
              const start_parameters_t &sParameters,
              const KalmanFitterOptions<outlier_finder_t> &kfOptions,
              KalmanFitterResult<source_link_t, parameters_t> &kfResult,
              const Surface *surfaceSequence = nullptr,
              size_t surfaceSequenceSize = 0) const {

    const bool IS_MAIN_THREAD = threadIdx.x == 0 && threadIdx.y == 0;

    // printf("Preparing %lu input measurements\n", sourcelinks.size());

    // Create the ActionList and AbortList
    using KalmanAborter = Aborter<source_link_t, parameters_t>;
    using KalmanActor = Actor<source_link_t, parameters_t>;

    // Create relevant options for the propagation options
    __shared__ PropagatorOptions<KalmanActor, KalmanAborter> kalmanOptions;
    __shared__ PropagatorResult propRes;

    if (IS_MAIN_THREAD) {
      kalmanOptions = PropagatorOptions<KalmanActor, KalmanAborter>(
          kfOptions.geoContext, kfOptions.magFieldContext);
      kalmanOptions.initializer.surfaceSequence = surfaceSequence;
      kalmanOptions.initializer.surfaceSequenceSize = surfaceSequenceSize;

      // Catch the actor and set the measurements
      kalmanOptions.action.inputMeasurements = std::move(sourcelinks);
      kalmanOptions.action.targetSurface = kfOptions.referenceSurface;

      // Set config for outlier finder
      kalmanOptions.action.m_outlierFinder = kfOptions.outlierFinder;
    }
    __syncthreads();

    // Run the fitter
    m_propagator.template propagate(sParameters, kalmanOptions, kfResult,
                                    propRes);

    if (!kfResult.result) {
      printf("KalmanFilter failed: \n");
      return false;
    }

    // Return the converted Track
    return true;
  }
#endif
};

} // namespace Acts
