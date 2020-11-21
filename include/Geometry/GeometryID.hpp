// This file is part of the Acts project.
//
// Copyright (C) 2016-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Geometry/GeometryStatics.hpp"

#include <cstdint>
#include <iosfwd>

namespace Acts {

/// Identifier for geometry nodes.
///
/// Each identifier can be split info the following components:
///
/// - Volumes                 - uses counting given by TrackingGeometry
/// - (Boundary)  Surfaces    - counts through boundary surfaces
/// - (Layer)     Surfaces    - counts confined layers
/// - (Approach)  Surfaces    - counts approach surfaces
/// - (Sensitive) Surfaces    - counts through sensitive surfaces
///
class GeometryID {
public:
  using Value = uint64_t;

  /// Construct from an already encoded value.
  constexpr GeometryID(Value encoded) : m_value(encoded) {}
  /// Construct default GeometryID with all values set to zero.
  GeometryID() = default;
  GeometryID(GeometryID &&) = default;
  GeometryID(const GeometryID &) = default;
  ~GeometryID() = default;
  GeometryID &operator=(GeometryID &&) = default;
  GeometryID &operator=(const GeometryID &) = default;

  /// Return the encoded value.
  ACTS_DEVICE_FUNC constexpr Value value() const { return m_value; }

  /// Return the volume identifier.
  ACTS_DEVICE_FUNC constexpr Value volume() const {
    return getBits(volume_mask);
  }
  /// Return the boundary identifier.
  ACTS_DEVICE_FUNC constexpr Value boundary() const {
    return getBits(boundary_mask);
  }
  /// Return the layer identifier.
  ACTS_DEVICE_FUNC constexpr Value layer() const { return getBits(layer_mask); }
  /// Return the approach identifier.
  ACTS_DEVICE_FUNC constexpr Value approach() const {
    return getBits(approach_mask);
  }
  /// Return the sensitive identifier.
  ACTS_DEVICE_FUNC constexpr Value sensitive() const {
    return getBits(sensitive_mask);
  }

  /// Set the volume identifier.
  ACTS_DEVICE_FUNC constexpr GeometryID &setVolume(Value volume) {
    return setBits(volume_mask, volume);
  }
  /// Set the boundary identifier.
  ACTS_DEVICE_FUNC constexpr GeometryID &setBoundary(Value boundary) {
    return setBits(boundary_mask, boundary);
  }
  /// Set the layer identifier.
  ACTS_DEVICE_FUNC constexpr GeometryID &setLayer(Value layer) {
    return setBits(layer_mask, layer);
  }
  /// Set the approach identifier.
  ACTS_DEVICE_FUNC constexpr GeometryID &setApproach(Value approach) {
    return setBits(approach_mask, approach);
  }
  /// Set the sensitive identifier.
  ACTS_DEVICE_FUNC constexpr GeometryID &setSensitive(Value sensitive) {
    return setBits(sensitive_mask, sensitive);
  }

private:
  Value m_value = 0;

  /// Extract the bit shift necessary to access the masked values.
  ACTS_DEVICE_FUNC static constexpr int extractShift(Value mask) {
    // use compiler builtin to extract the number of trailing bits from the
    // mask. the builtin should be available on all supported compilers.
    // need unsigned long long version (...ll) to ensure uint64_t compatibility.
    // WARNING undefined behaviour for mask == 0 which we should not have.
    //	  return __builtin_ctzll(mask);
    switch (mask) {
    case volume_mask:
      return 56;
    case boundary_mask:
      return 48;
    case layer_mask:
      return 36;
    case approach_mask:
      return 28;
    case sensitive_mask:
      return 0;
    default:
      return 0;
    }
  }

  /// Extract the masked bits from the encoded value.
  ACTS_DEVICE_FUNC constexpr Value getBits(Value mask) const {
     printf("getBits\n"); 
	  return (m_value & mask) >> extractShift(mask);
  }
  /// Set the masked bits to id in the encoded value.
  ACTS_DEVICE_FUNC constexpr GeometryID &setBits(Value mask, Value id) {
    m_value = (m_value & ~mask) | ((id << extractShift(mask)) & mask);
    // return *this here so we need to write less lines in the set... methods
    return *this;
  }

  ACTS_DEVICE_FUNC friend constexpr bool operator==(GeometryID lhs,
                                                    GeometryID rhs) {
    return lhs.m_value == rhs.m_value;
  }
  ACTS_DEVICE_FUNC friend constexpr bool operator<(GeometryID lhs,
                                                   GeometryID rhs) {
    return lhs.m_value < rhs.m_value;
  }
};

} // namespace Acts
