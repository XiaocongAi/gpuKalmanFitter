// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Surfaces/Surface.hpp"
#include "Utilities/Definitions.hpp"

#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>

namespace Acts {
/// DirectNavigator Actor struct, called Initializer
///
/// This is needed for the initialization of the
/// surface sequence
struct DirectNavigatorInitializer {
  /// The Surface sequence
  const Surface *surfaceSequence = nullptr;

  /// The surface sequence size
  size_t surfaceSequenceSize = 0;

  /// Actor result / state
  struct this_result {
    bool initialized = false;
  };
  using result_type = this_result;

  /// Defaulting the constructor
  DirectNavigatorInitializer() = default;

  /// Actor operator call
  /// @tparam statet Type of the full propagator state
  /// @tparam stepper_t Type of the stepper
  ///
  /// @param state the entire propagator state
  /// @param r the result of this Actor
  template <typename propagator_state_t, typename stepper_t>
  ACTS_DEVICE_FUNC void operator()(propagator_state_t &state,
                                   const stepper_t & /*unused*/,
                                   result_type &r) const {
    // Only act once
    if (not r.initialized) {
      // Initialize the surface sequence
      state.navigation.surfaceSequence = surfaceSequence;
      state.navigation.surfaceSequenceSize = surfaceSequenceSize;
      r.initialized = true;
    }
  }

  /// Actor operator call - resultless, unused
  template <typename propagator_state_t, typename stepper_t>
  ACTS_DEVICE_FUNC void operator()(propagator_state_t & /*unused*/,
                                   const stepper_t & /*unused*/) const {}
};

/// DirectNavigator class
///
/// This is a fully guided navigator that progresses through
/// a pre-given sequence of surfaces.
///
/// This can either be used as a validation tool, for truth
/// tracking, or track refitting
template <typename surface_derived_t = PlaneSurface<InfiniteBounds>>
class DirectNavigator {
public:
  /// The sequentially crossed surfaces
  // using SurfaceSequence = std::vector<const Surface *>;
  // using SurfaceIter = std::vector<const Surface *>::iterator;
  using SurfaceIter = unsigned int;

  /// Defaulted Constructed
  DirectNavigator() = default;

  /// The tolerance used to define "surface reached"
  double tolerance = s_onSurfaceTolerance;

  /// Nested State struct
  ///
  /// It acts as an internal state which is
  /// created for every propagation/extrapolation step
  /// and keep thread-local navigation information
  struct State {
    /// Externally provided surfaces - expected to be ordered
    /// along the path
    const Surface *surfaceSequence = nullptr;

    /// The surface sequence size
    size_t surfaceSequenceSize = 0;

    /// Iterator the the next surface
    SurfaceIter nextSurfaceIter = 0;

    /// Navigation state - external interface: the start surface
    const Surface *startSurface = nullptr;
    /// Navigation state - external interface: the current surface
    const Surface *currentSurface = nullptr;
    /// Navigation state - external interface: the target surface
    const Surface *targetSurface = nullptr;

    /// Navigation state - external interface: target is reached
    bool targetReached = false;
    /// Navigation state - external interface: a break has been detected
    bool navigationBreak = false;
  };

  template <typename T>
  static ACTS_DEVICE_FUNC void *advanceByDatatype(void *addr) {
    // return static_cast<T*>(addr) + sizeof(T);
    return addr + sizeof(T);
  }

  template <typename T>
  static ACTS_DEVICE_FUNC const void *advanceByDatatype(const void *addr) {
    // return static_cast<const T*>(addr) + sizeof(T);
    return addr + sizeof(T);
  }

  static ACTS_DEVICE_FUNC Surface *advanceSurfacePtr(Surface *ptr) {
    //  switch (ptr->type()) {
    //  case Surface::Plane:
    return (Surface *)advanceByDatatype<surface_derived_t>(ptr);
    //   default:
    //    return ptr + 1;
    //  }
  }

  static ACTS_DEVICE_FUNC const Surface *advanceSurfacePtr(const Surface *ptr) {
    //  switch (ptr->type()) {
    //  case Surface::Plane:
    return (Surface const *)advanceByDatatype<surface_derived_t>(ptr);
    //  default:
    //    return ptr + 1;
    //  }
  }

  /// @brief Navigator status call
  ///
  /// @tparam propagator_state_t is the type of Propagatgor state
  /// @tparam stepper_t is the used type of the Stepper by the Propagator
  ///
  /// @param [in,out] state is the mutable propagator state object
  /// @param [in] stepper Stepper in use
  template <typename propagator_state_t, typename stepper_t>
  ACTS_DEVICE_FUNC void status(propagator_state_t &state,
                               const stepper_t &stepper) const {

    // Navigator status always resets the current surface
    state.navigation.currentSurface = nullptr;
    // Check if we are on surface
    if (state.navigation.nextSurfaceIter !=
        state.navigation.surfaceSequenceSize) {
      auto *surfacePtr = state.navigation.surfaceSequence;
      for (int i = 0; i < state.navigation.nextSurfaceIter; ++i) {
        surfacePtr = advanceSurfacePtr(surfacePtr);
      }
      // Establish the surface status
      auto surfaceStatus =
          stepper.template updateSurfaceStatus<surface_derived_t>(
              state.stepping, *surfacePtr, false);
      if (surfaceStatus == Intersection::Status::onSurface) {
        // Set the current surface
        state.navigation.currentSurface = surfacePtr;
        // Move the sequence to the next surface
        ++state.navigation.nextSurfaceIter;
        //@Todo: I guess this could be removed as it's already done in the
        // updateSurfaceStatus
        if (state.navigation.nextSurfaceIter !=
            state.navigation.surfaceSequenceSize) {
          stepper.releaseStepSize(state.stepping);
        }
      }
    }
  }

  /// @brief Navigator target call
  ///
  /// @tparam propagator_state_t is the type of Propagatgor state
  /// @tparam stepper_t is the used type of the Stepper by the Propagator
  ///
  /// @param [in,out] state is the mutable propagator state object
  /// @param [in] stepper Stepper in use
  template <typename propagator_state_t, typename stepper_t>
  ACTS_DEVICE_FUNC void target(propagator_state_t &state,
                               const stepper_t &stepper) const {

    // Navigator target always resets the current surface
    state.navigation.currentSurface = nullptr;
    if (state.navigation.nextSurfaceIter !=
        state.navigation.surfaceSequenceSize) {
      auto *surfacePtr = state.navigation.surfaceSequence;
      for (int i = 0; i < state.navigation.nextSurfaceIter; ++i) {
        surfacePtr = advanceSurfacePtr(surfacePtr);
      }
      // Establish & update the surface status
      auto surfaceStatus =
          stepper.template updateSurfaceStatus<surface_derived_t>(
              state.stepping, *surfacePtr, false);
      if (surfaceStatus == Intersection::Status::unreachable) {
        // Move the sequence to the next surface
        ++state.navigation.nextSurfaceIter;
      }
    } else {
      // Set the navigation break
      state.navigation.navigationBreak = true;
      // If no externally provided target is given, the target is reached
      if (state.navigation.targetSurface == nullptr) {
        state.navigation.targetReached = true;
      }
    }
  }
};

} // namespace Acts
