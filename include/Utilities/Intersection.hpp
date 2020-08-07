// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// Intersection.hpp, Acts project
///////////////////////////////////////////////////////////////////

#pragma once
#include "Definitions.hpp"
#include <limits>

namespace Acts {

///  @struct Intersection
///
///  Intersection struct used for position
struct Intersection {
  /// Nested Status enum
  enum class Status : int {
    missed = 0,
    unreachable = 0,
    reachable = 1,
    onSurface = 2
  };

  /// Position of the intersection
  Vector3D position{0., 0., 0.};
  /// Signed path length to the intersection (if valid)
  double pathLength{std::numeric_limits<double>::infinity()};
  /// The Status of the intersection
  Status status{Status::unreachable};

  /// Constructor with arguments
  ///
  /// @param sinter is the position of the intersection
  /// @param slength is the path length to the intersection
  /// @param svalid is a boolean indicating if intersection is valid
  ACTS_DEVICE_FUNC Intersection(const Vector3D &sinter, double slength,
                                Status sstatus)
      : position(sinter), pathLength(slength), status(sstatus) {}

  /// Default constructor
  Intersection() = default;

  /// Bool() operator for validity checking
  ACTS_DEVICE_FUNC explicit operator bool() const {
    return (status != Status::missed);
  }

  /// Smaller operator for sorting,
  /// - it respects the validity of the intersection
  /// @param si is the intersection for testing
  ACTS_DEVICE_FUNC bool operator<(const Intersection &si) const {
    if (status == Status::unreachable) {
      return false;
    }
    // Now check the pathLength
    if (si.status != Status::unreachable) {
      return (pathLength < si.pathLength);
    }
    // The current one wins, no re-ordering
    return true;
  }

  /// Greater operator for sorting,
  /// - it respects the validity of the intersection
  /// @param si is the intersection for testing
  ACTS_DEVICE_FUNC bool operator>(const Intersection &si) const {
    if (status == Status::unreachable) {
      return false;
    }
    // Now check the pathLength
    if (si.status != Status::unreachable) {
      return (pathLength > si.pathLength);
    }
    // The current one wins, no re-ordering
    return true;
  }
};

/// @brief class extensions to return also the object and a representation
template <typename object_t, typename representation_t = object_t>
class ObjectIntersection {
public:
  /// The intersection itself
  Intersection intersection{};
  /// The object that was (tried to be) intersected
  const object_t *object{nullptr};
  /// The representation of this object
  const representation_t *representation{nullptr};

  /// The alternative intersections
  Intersection alternative{};

  /// Default constructor
  ObjectIntersection() = default;

  /// Object intersection - symmetric setup
  ///
  /// @param sInter is the intersection
  /// @param sObject is the object to be instersected
  /// @param sRepresentation is the object represenatation
  template <typename T = representation_t,
            std::enable_if_t<std::is_same<T, object_t>::value, int> = 0>
  ACTS_DEVICE_FUNC ObjectIntersection(const Intersection &sInter,
                                      const object_t *sObject)
      : intersection(sInter), object(sObject), representation(sObject) {}

  /// Object intersection
  ///
  /// @param sInter is the intersection
  /// @param sObject is the object to be instersected
  /// @param sRepresentation is the object represenatation
  ACTS_DEVICE_FUNC ObjectIntersection(const Intersection &sInter,
                                      const object_t *sObject,
                                      const representation_t *sRepresentation)
      : intersection(sInter), object(sObject), representation(sRepresentation) {
  }

  /// Bool() operator for validity checking
  ACTS_DEVICE_FUNC explicit operator bool() const { return bool(intersection); }

  /// @brief Smaller operator for ordering & sorting
  ///
  /// This operator will ignore the alternative, but simply
  /// order the representing intersection
  ///
  /// @param oi is the source intersection for comparison
  ACTS_DEVICE_FUNC bool
  operator<(const ObjectIntersection<object_t, representation_t> &oi) const {
    return (intersection < oi.intersection);
  }

  /// @brief Greater operator for ordering & sorting
  ///
  /// This operator will ignore the alternative, but simply
  /// order the representing intersection
  ///
  /// @param oi is the source intersection for comparison
  ACTS_DEVICE_FUNC bool
  operator>(const ObjectIntersection<object_t, representation_t> &oi) const {
    return (intersection > oi.intersection);
  }
};

struct SameSurfaceIntersection {
  /// @brief comparison operator
  ///
  /// This is a struct to pick out intersection with identical surfaces
  ///
  /// @tparam intersection_t Type of the intersection object
  /// @param i1 First intersection to test
  /// @param i2 Second intersection to test
  template <typename intersection_t>
  ACTS_DEVICE_FUNC bool operator()(const intersection_t &i1,
                                   const intersection_t &i2) const {
    return (i1.object == i2.object);
  }
};

} // namespace Acts
