// This file is part of the Acts project.
//
// Copyright (C) 2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Utilities/ParameterDefinitions.hpp"

#include <optional>

namespace Acts {

/// @enum TrackStateFlag
///
/// This enum describes the type of TrackState
enum TrackStateFlag {
  MeasurementFlag = 0,
  ParameterFlag = 1,
  OutlierFlag = 2,
  HoleFlag = 3,
  MaterialFlag = 4,
  NumTrackStateFlags = 5
};

using TrackStateType = std::bitset<TrackStateFlag::NumTrackStateFlags>;

class Surface;

/// @class TrackState
///
/// @brief Templated class to hold the track information
/// on a surface along the trajectory
///
/// @tparam source_link_t Type of the source link
/// @tparam parameters_t Type of the parameters on the surface
///
/// @note the Surface is only stored as a pointer, i.e. it is
/// assumed the surface lives longer than the TrackState
template <typename source_link_t, typename parameters_t> class TrackState {

public:
  using SourceLink = source_link_t;
  using Parameters = parameters_t;
  using Jacobian = typename Parameters::CovarianceMatrix;

  // TrackState() = delete;
  TrackState() = default;

  /// Constructor from (uncalibrated) measurement
  ///
  /// @param m The measurement object
  ACTS_DEVICE_FUNC TrackState(SourceLink m) : m_surface(&m.referenceSurface()) {
    measurement.uncalibrated = std::move(m);
    // m_typeFlags.set(MeasurementFlag);
  }

  /// Constructor from parameters
  ///
  /// @tparam parameters_t Type of the predicted parameters
  /// @param p The parameters object
  TrackState(parameters_t p) {
    m_surface = &p.referenceSurface();
    parameter.predicted = std::move(p);
    m_typeFlags.set(ParameterFlag);
  }

  /// Virtual destructor
  virtual ~TrackState() = default;

  /// Copy constructor
  ///
  /// @param rhs is the source TrackState
  ACTS_DEVICE_FUNC TrackState(const TrackState &rhs)
      : parameter(rhs.parameter), measurement(rhs.measurement),
        m_surface(rhs.m_surface), m_typeFlags(rhs.m_typeFlags) {}

  /// Copy move constructor
  ///
  /// @param rhs is the source TrackState
  ACTS_DEVICE_FUNC TrackState(TrackState &&rhs)
      : parameter(std::move(rhs.parameter)),
        measurement(std::move(rhs.measurement)),
        m_surface(std::move(rhs.m_surface)),
        m_typeFlags(std::move(rhs.m_typeFlags)) {}

  /// Assignment operator
  ///
  /// @param rhs is the source TrackState
  ACTS_DEVICE_FUNC TrackState &operator=(const TrackState &rhs) {
    parameter = rhs.parameter;
    measurement = rhs.measurement;
    m_surface = rhs.m_surface;
    m_typeFlags = rhs.m_typeFlags;
    return (*this);
  }

  /// Assignment move operator
  ///
  /// @param rhs is the source TrackState
  ACTS_DEVICE_FUNC TrackState &operator=(TrackState &&rhs) {
    parameter = std::move(rhs.parameter);
    measurement = std::move(rhs.measurement);
    m_surface = std::move(rhs.m_surface);
    m_typeFlags = std::move(rhs.m_typeFlags);
    return (*this);
  }

  /// @brief return method for the surface
  ACTS_DEVICE_FUNC const Surface &referenceSurface() const {
    return (*m_surface);
  }

  /// @brief set the type flag
  ACTS_DEVICE_FUNC void setType(const TrackStateFlag &flag,
                                bool status = true) {
    m_typeFlags.set(flag, status);
  }

  /// @brief test if the tracks state is flagged as a given type
  ACTS_DEVICE_FUNC bool isType(const TrackStateFlag &flag) const {
    assert(flag < NumTrackStateFlags);
    return m_typeFlags.test(flag);
  }

  /// @brief return method for the type flags
  ACTS_DEVICE_FUNC TrackStateType type() const { return m_typeFlags; }

  /// @brief number of Measured parameters, forwarded
  /// @note This only returns a value if there is a calibrated measurement
  ///       set. If not, this returns std::nullopt
  ///
  /// @return number of measured parameters, or std::nullopt
  /// std::optional<size_t> size() {
  ///   if (this->measurement.calibrated) {
  ///     return MeasurementHelpers::getSize(*this->measurement.calibrated);
  ///   }
  ///   return std::nullopt;
  /// }

  /// The parameter part
  /// This is all the information that concerns the
  /// the track parameterisation and the jacobian
  /// It is enough to to run the track smoothing
  struct {
    /// The predicted state
    Parameters predicted;
    /// The filtered state
    Parameters filtered;
    /// The smoothed state
    Parameters smoothed;
    /// The transport jacobian matrix
    Jacobian jacobian;
    /// The path length along the track - will help sorting
    double pathLength = 0.;
    /// chisquare
    double chi2 = 0;
  } parameter;

  /// @brief Nested measurement part
  /// This is the unalibrated and calibrated measurement
  /// (in case the latter is different)
  struct {
    /// The optional (uncalibrated) measurement
    SourceLink uncalibrated;
  } measurement;

private:
  /// The surface of this TrackState
  const Surface *m_surface = nullptr;
  /// The type flag of this TrackState
  TrackStateType m_typeFlags;
};
} // namespace Acts
