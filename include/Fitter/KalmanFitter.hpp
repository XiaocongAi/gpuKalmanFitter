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
#include "Propagator/Propagator.hpp"
#include "Propagator/StandardAborters.hpp"
#include "Propagator/detail/CovarianceEngine.hpp"
#include "Propagator/detail/PointwiseMaterialInteraction.hpp"
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
                      const MagneticFieldContext &mctx, bool rsmoothing = true,
                      const OutlierFinder &outlierFinder_ = VoidOutlierFinder(),
                      const Surface *rSurface = nullptr,
                      bool mScattering = true, bool eLoss = true)
      : geoContext(gctx), magFieldContext(mctx), smoothing(rsmoothing),
        outlierFinder(outlierFinder_), referenceSurface(rSurface),
        multipleScattering(mScattering), energyLoss(eLoss) {}

  /// Context object for the geometry
  GeometryContext geoContext;
  /// Context object for the magnetic field
  MagneticFieldContext magFieldContext;

  /// The config for the outlier finder
  OutlierFinder outlierFinder;

  /// The reference Surface
  const Surface *referenceSurface = nullptr;

  /// Whether to run smoothing
  bool smoothing = true;

  /// Whether to consider multiple scattering
  bool multipleScattering = true;

  /// Whether to consider energy loss
  bool energyLoss = true;
};

template <typename source_link_t, typename parameters_t,
          typename target_surface_t>
struct KalmanFitterResult {
  using TrackStateType = TrackState<source_link_t, parameters_t>;

  // Fitted states that the actor has handled.
  CudaKernelContainer<TrackStateType> fittedStates;

  // The optional Parameters at the provided surface
  BoundParameters<target_surface_t> fittedParameters;

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
///
/// The Kalman filter contains an Actor and a Sequencer sub-class.
/// The Sequencer has to be part of the Navigator of the Propagator
/// in order to initialize and provide the measurement surfaces.
///
/// The Actor is part of the Propagation call and does the Kalman update
/// and eventually the smoothing.  Updater, Smoother are
/// given to the Actor for further use:
/// - The Updater is the implemented kalman updater formalism, it
///   runs via a visitor pattern through the measurements.
/// - The Smoother is called at the end of the forward fit by the Actor.
/// - The outlier finder is called during the filtering by the Actor.
///   It determines if the measurement is an outlier
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
  using NavigationSurface = typename propagator_t::NavigationSurface;

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
  template <typename source_link_t, typename parameters_t,
            typename target_surface_t>
  class Actor {
  public:
    // Broadcast the result type
    using result_type =
        KalmanFitterResult<source_link_t, parameters_t, target_surface_t>;

    // Broadcast the track state type
    using TrackStateType = typename result_type::TrackStateType;

    /// Broadcast the input measurement container type
    using InputMeasurementsType = CudaKernelContainer<source_link_t>;
    // using InputMeasurementsType = std::vector<source_link_t>;

    /// The target surface
    const Surface *targetSurface = nullptr;

    /// Allows retrieving measurements for a surface
    InputMeasurementsType inputMeasurements;

    /// Whether to run smoothing
    bool smoothing = true;

    /// Whether to consider multiple scattering.
    bool multipleScattering = true;

    /// Whether to consider energy loss.
    bool energyLoss = true;

    /// Add constructor with updater and smoother
    ACTS_DEVICE_FUNC Actor(updater_t pUpdater = updater_t(),
                           smoother_t pSmoother = smoother_t())
        : m_updater(std::move(pUpdater)), m_smoother(std::move(pSmoother)) {}

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
      if ((result.measurementStates == inputMeasurements.size() or
           (result.measurementStates > 0 and
            state.navigation.navigationBreak)) and
          !result.smoothed and !result.finished) {
        if (not smoothing) {
          printf("Finish without smoothing\n");
          result.finished = true;
        } else {
          // printf("Finalize/run smoothing\n");
          auto res = finalize(state, stepper, result);
          if (!res) {
            printf("Error in finalize:\n");
            result.result = false;
          }
        }
      }

      // Post-finalization:
      // - Progress to target/reference surface and built the final track
      // parameters
      if (result.smoothed and !result.finished and
          targetReached.
          operator()<propagator_state_t, stepper_t, target_surface_t>(
              state, stepper, *targetSurface)) {
        // printf("Completing\n");
        // Construct a tempory jacobian and path, which is necessary for calling
        // the boundState
        typename TrackStateType::Jacobian jac;
        double path;
        // Transport & bind the parameter to the final surface
        stepper.template boundState<target_surface_t>(
            state.stepping, *targetSurface, result.fittedParameters, jac, path);

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
      auto sourcelink_it = inputMeasurements.find_if(sFinder);
      // No source link, still return true
      if (sourcelink_it == inputMeasurements.end()) {
        return true;
      }
      // Screen out the source link
      // auto pos = (*sourcelink_it).globalPosition(state.options.geoContext);
      // printf("sl position = (%f, %f, %f)\n", pos.x(), pos.y(), pos.z());

      // create track state on the vector from sourcelink
      result.fittedStates[result.measurementStates] =
          TrackStateType(*sourcelink_it);
      TrackStateType &trackState =
          result.fittedStates[result.measurementStates];

      // Transport & bind the state to the current surface
      stepper.template boundState<NavigationSurface>(
          state.stepping, *surface, trackState.parameter.predicted,
          trackState.parameter.jacobian, trackState.parameter.pathLength);

      // Screen out the predicted parameters
      // auto prePos = trackState.parameter.predicted.position();
      // printf("Predicted parameter position = (%f, %f, %f)\n", prePos.x(),
      // prePos.y(), prePos.z());

      // If the update is successful, set covariance and
      auto updateRes = m_updater(state.options.geoContext, trackState);
      if (!updateRes) {
        // printf("Update step failed:\n");
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

      // The material effects after the filtering
      materialInteractor(surface, state, stepper, fullUpdate);

      // We count the state with measurement
      ++result.measurementStates;
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
          m_smoother(state.options.geoContext, result.fittedStates);

      if (smoothedPars) {
        // printf("pos=%d,%d\n",(*smoothedPars).position().x(),
        // (*smoothedPars).position().y());
        const auto freeParams =
            detail::coordinate_transformation::boundParameters2freeParameters<
                NavigationSurface>(state.options.geoContext,
                                   smoothedPars->parameters(),
                                   smoothedPars->referenceSurface());
        stepper.update(state.stepping, freeParams,
                       *(smoothedPars->covariance()));

        // Reverse the propagation direction
        state.stepping.navDir = backward;
        state.stepping.stepSize = ConstrainedStep(
            state.stepping.navDir * std::abs(state.options.maxStepSize));
        // Set accumulatd path to zero before targeting surface
        state.stepping.pathAccumulated = 0.;

        return true;
      }

      return false;
    }

    /// @brief Kalman actor operation : material interaction
    ///
    /// @tparam propagator_state_t is the type of Propagagor state
    /// @tparam stepper_t Type of the stepper
    ///
    /// @param surface The surface where the material interaction happens
    /// @param state The mutable propagator state object
    /// @param stepper The stepper in use
    /// @param updateStage The materal update stage
    ///
    template <typename propagator_state_t, typename stepper_t>
    ACTS_DEVICE_FUNC void materialInteractor(
        const Surface *surface, propagator_state_t &state, stepper_t &stepper,
        const MaterialUpdateStage &updateStage = fullUpdate) const {
      // Indicator if having material
      bool hasMaterial = false;

      // The material might be zero
      if (surface and surface->surfaceMaterial().materialSlab()) {
        // Prepare relevant input particle properties
        detail::PointwiseMaterialInteraction interaction(surface, state,
                                                         stepper);
        // Evaluate the material properties
        if (interaction.evaluateMaterialSlab(state, updateStage)) {
          // Surface has material at this stage
          hasMaterial = true;

          // Evaluate the material effects
          interaction.evaluatePointwiseMaterialInteraction(multipleScattering,
                                                           energyLoss);
          // Screen out material effects info
          // ACTS_VERBOSE("Material effects on surface: "
          //             << surface->geometryId()
          //             << " at update stage: " << updateStage << " are :");
          // ACTS_VERBOSE("eLoss = "
          //             << interaction.Eloss << ", "
          //             << "variancePhi = " << interaction.variancePhi << ", "
          //             << "varianceTheta = " << interaction.varianceTheta
          //             << ", "
          //             << "varianceQoverP = " << interaction.varianceQoverP);

          // Update the state and stepper with material effects
          interaction.updateState(state, stepper);
        }
      }

      // if (not hasMaterial) {
      // Screen out message
      // ACTS_VERBOSE("No material effects on surface: " <<
      // surface->geometryId()
      //                                                << " at update stage:
      //                                                "
      //                                                << updateStage);
      //}
    }

#ifdef __CUDACC__

    /// @brief Kalman actor operation with multiple threads on Device
    ///
    /// @tparam propagator_state_t is the type of Propagagor state
    /// @tparam stepper_t Type of the stepper
    ///
    /// @param state is the mutable propagator state object
    /// @param stepper The stepper in use
    /// @param result is the mutable result state object
    template <typename propagator_state_t, typename stepper_t>
    __device__ void actionOnDevice(propagator_state_t &state,
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

      //      bool IS_MAIN_THREAD = true;
      //      #ifdef __CUDA_ARCH__
      //      	IS_MAIN_THREAD = threadIdx.x == 0 && threadIdx.y == 0;
      //      #endif

      // Finalization:
      // when all track states have been handled or the navigation is breaked,
      // reset navigation&stepping before run backward filtering or
      // proceed to run smoothing
      if ((result.measurementStates == inputMeasurements.size() or
           (result.measurementStates > 0 and
            state.navigation.navigationBreak)) and
          !result.smoothed and !result.finished) {
        if (not smoothing) {
          //printf("Finish without smoothing\n");
          result.finished = true;
        } else {
          // printf("Finalize/run smoothing\n");
          auto res = finalize(state, stepper, result);
          if (!res) {
            printf("Error in finalize:\n");
            result.result = false;
          }
        }
      }

      // Post-finalization:
      // - Progress to target/reference surface and built the final track
      // parameters
      if (result.smoothed and !result.finished and
          targetReached.
          operator()<propagator_state_t, stepper_t, target_surface_t>(
              state, stepper, *targetSurface)) {
        // printf("Completing\n");
        // Construct a tempory jacobian and path, which is necessary for calling
        // the boundState
        typename TrackStateType::Jacobian jac;
        double path;
        // Transport & bind the parameter to the final surface
        stepper.template boundState<target_surface_t>(
            state.stepping, *targetSurface, result.fittedParameters, jac, path);

        // Remember the track fitting is done
        result.finished = true;
      }
    }

    /// @brief Kalman actor operation : update on device with multiple threads
    /// on device
    ///
    /// @tparam propagator_state_t is the type of Propagagor state
    /// @tparam stepper_t Type of the stepper
    ///
    /// @param surface The surface where the update happens
    /// @param state The mutable propagator state object
    /// @param stepper The stepper in use
    /// @param result The mutable result state object
    template <typename propagator_state_t, typename stepper_t>
    __device__ bool
    filterOnDevice(const Surface *surface, propagator_state_t &state,
                   const stepper_t &stepper, result_type &result) const {
      // Try to find the surface in the measurement surfaces
      SurfaceFinder sFinder{surface};
      auto sourcelink_it = inputMeasurements.find_if(sFinder);
      // No source link, still return true
      if (sourcelink_it == inputMeasurements.end()) {
        return true;
      }
      // Screen out the source link
      // auto pos = (*sourcelink_it).globalPosition(state.options.geoContext);
      // printf("sl position = (%f, %f, %f)\n", pos.x(), pos.y(), pos.z());

      // create track state on the vector from sourcelink
      result.fittedStates[result.measurementStates] =
          TrackStateType(*sourcelink_it);
      TrackStateType &trackState =
          result.fittedStates[result.measurementStates];

      // Transport & bind the state to the current surface
      // @todo: to be changed to boundStateOnDevice
      stepper.template boundState<NavigationSurface>(
          state.stepping, *surface, trackState.parameter.predicted,
          trackState.parameter.jacobian, trackState.parameter.pathLength);

      // Screen out the predicted parameters
      // auto prePos = trackState.parameter.predicted.position();
      // printf("Predicted parameter position = (%f, %f, %f)\n", prePos.x(),
      // prePos.y(), prePos.z());

      // If the update is successful, set covariance and
      auto updateRes = m_updater(state.options.geoContext, trackState);
      if (!updateRes) {
        // printf("Update step failed:\n");
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

      // The material effects after the filtering
      materialInteractor(surface, state, stepper, fullUpdate);

      // We count the state with measurement
      ++result.measurementStates;
      return true;
    }

    /// @brief Kalman actor operation : finalize with multiple threads on device
    ///
    /// @tparam propagator_state_t is the type of Propagagor state
    /// @tparam stepper_t Type of the stepper
    ///
    /// @param state is the mutable propagator state object
    /// @param stepper The stepper in use
    /// @param result is the mutable result state object
    template <typename propagator_state_t, typename stepper_t>
    __device__ bool finalizeOnDevice(propagator_state_t &state,
                                     const stepper_t &stepper,
                                     result_type &result) const {
      // Remember you smoothed the track states
      result.smoothed = true;

      // Smooth the track states
      const auto &smoothedPars =
          m_smoother(state.options.geoContext, result.fittedStates);

      if (smoothedPars) {
        // printf("pos=%d,%d\n",(*smoothedPars).position().x(),
        // (*smoothedPars).position().y());
        const auto freeParams =
            detail::coordinate_transformation::boundParameters2freeParameters<
                NavigationSurface>(state.options.geoContext,
                                   smoothedPars->parameters(),
                                   smoothedPars->referenceSurface());
        stepper.update(state.stepping, freeParams,
                       *(smoothedPars->covariance()));

        // Reverse the propagation direction
        state.stepping.navDir = backward;
        state.stepping.stepSize = ConstrainedStep(
            state.stepping.navDir * std::abs(state.options.maxStepSize));
        // Set accumulatd path to zero before targeting surface
        state.stepping.pathAccumulated = 0.;

        return true;
      }
      return false;
    }

#endif

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
    template <typename propagator_state_t, typename stepper_t,
              typename result_type>
    ACTS_DEVICE_FUNC bool operator()(propagator_state_t & /*state*/,
                                     const stepper_t & /*stepper*/,
                                     const result_type &result) const {
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
  ///
  /// @return the output as an output track
  template <typename source_link_t, typename start_parameters_t,
            typename parameters_t = BoundParameters<NavigationSurface>,
            typename target_surface_t = PlaneSurface<InfiniteBounds>>
  ACTS_DEVICE_FUNC bool
  fit(const CudaKernelContainer<source_link_t> &sourcelinks,
      const start_parameters_t &sParameters,
      const KalmanFitterOptions<outlier_finder_t> &kfOptions,
      KalmanFitterResult<source_link_t, parameters_t, target_surface_t>
          &kfResult,
      const Surface *surfaceSequence = nullptr,
      size_t surfaceSequenceSize = 0) const {

    PUSH_RANGE("fit", 0);

    // printf("Preparing %lu input measurements\n", sourcelinks.size());
    // Create the ActionList and AbortList
    using KalmanAborter = Aborter<source_link_t, parameters_t>;
    using KalmanActor = Actor<source_link_t, parameters_t, target_surface_t>;

    // Create relevant options for the propagation options
    PropagatorOptions<KalmanActor, KalmanAborter> kalmanOptions(
        kfOptions.geoContext, kfOptions.magFieldContext);
    kalmanOptions.initializer.surfaceSequence = surfaceSequence;
    kalmanOptions.initializer.surfaceSequenceSize = surfaceSequenceSize;
    // Tells the direct navigator that the target surface is not empty
    kalmanOptions.initializer.targetSurface = kfOptions.referenceSurface;

    // Catch the actor and set the measurements
    auto &kalmanActor = kalmanOptions.action;
    kalmanActor.inputMeasurements = std::move(sourcelinks);
    kalmanActor.targetSurface = kfOptions.referenceSurface;
    kalmanActor.multipleScattering = kfOptions.multipleScattering;
    kalmanActor.energyLoss = kfOptions.energyLoss;
    kalmanActor.smoothing = kfOptions.smoothing;

    // Set config for outlier finder
    kalmanActor.m_outlierFinder = kfOptions.outlierFinder;

    // Run the fitter
    const auto propRes =
        m_propagator.template propagate(sParameters, kalmanOptions, kfResult);

    POP_RANGE();

    if (!kfResult.result or (kfResult.result and kfResult.measurementStates !=
                                                     surfaceSequenceSize)) {
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
  ///
  /// @return the output as an output track
  template <typename source_link_t, typename start_parameters_t,
            typename parameters_t = BoundParameters<NavigationSurface>,
            typename target_surface_t = PlaneSurface<InfiniteBounds>>
  __device__ void
  fitOnDevice(const CudaKernelContainer<source_link_t> &sourcelinks,
              const start_parameters_t &sParameters,
              const KalmanFitterOptions<outlier_finder_t> &kfOptions,
              KalmanFitterResult<source_link_t, parameters_t, target_surface_t>
                  &kfResult,
              bool &status, const Surface *surfaceSequence = nullptr,
              size_t surfaceSequenceSize = 0) const {

    const bool IS_MAIN_THREAD = threadIdx.x == 0 && threadIdx.y == 0;

    // printf("Preparing %lu input measurements\n", sourcelinks.size());

    // Create the ActionList and AbortList
    using KalmanAborter = Aborter<source_link_t, parameters_t>;
    using KalmanActor = Actor<source_link_t, parameters_t, target_surface_t>;

    // Create relevant options for the propagation options
    __shared__ PropagatorOptions<KalmanActor, KalmanAborter> kalmanOptions;
    __shared__ PropagatorResult propRes;

    if (IS_MAIN_THREAD) {
      kalmanOptions = PropagatorOptions<KalmanActor, KalmanAborter>(
          kfOptions.geoContext, kfOptions.magFieldContext);
      kalmanOptions.initializer.surfaceSequence = surfaceSequence;
      kalmanOptions.initializer.surfaceSequenceSize = surfaceSequenceSize;
      // Tells the direct navigator that the target surface is not empty
      kalmanOptions.initializer.targetSurface = kfOptions.referenceSurface;

      // Catch the actor and set the measurements
      kalmanOptions.action.inputMeasurements = std::move(sourcelinks);
      kalmanOptions.action.targetSurface = kfOptions.referenceSurface;
      kalmanOptions.action.multipleScattering = kfOptions.multipleScattering;
      kalmanOptions.action.energyLoss = kfOptions.energyLoss;
      kalmanOptions.action.smoothing = kfOptions.smoothing;

      // Set config for outlier finder
      kalmanOptions.action.m_outlierFinder = kfOptions.outlierFinder;

      propRes = PropagatorResult();
    }
    __syncthreads();

    // Run the fitter
    m_propagator.template propagate(sParameters, kalmanOptions, kfResult,
                                    propRes);

    // update the fit status with the main thread
    if (IS_MAIN_THREAD) {
      status = true;
      if (!kfResult.result or (kfResult.result and kfResult.measurementStates !=
                                                       surfaceSequenceSize)) {
        printf("KalmanFilter failed: \n");
        status = false;
      }
    }
    __syncthreads();
  }
#endif
};

} // namespace Acts
