// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Utilities/Definitions.hpp"
#include "Utilities/ParameterTypes.hpp"

#include <cmath>
#include <type_traits>

// The user can override the (track) parameter ordering and underlying scalar
// type. If the variable is defined, it must point to a header file that
// contains the same enum and type definitions for bound and free track
// parameters as well as space points as given below.
#ifdef ACTS_PARAMETER_DEFINITIONS_HEADER
#include ACTS_PARAMETER_DEFINITIONS_HEADER
#else
namespace Acts {

// Note:
// The named indices are use to access raw data vectors and matrices at the
// lowest level. Since the interpretation of some of the components, e.g. local
// position and the inverse-momentum-like component, depend on additional
// information the names have some ambiguity. This can only be resolved at a
// higher logical level and no attempt is made to resolve it here.

/// Components of a bound track parameters vector.
///
/// To be used to access components by named indices instead of just numbers.
/// This must be a regular `enum` and not a scoped `enum class` to allow
/// implicit conversion to an integer. The enum value are thus visible directly
/// in `namespace Acts` and are prefixed to avoid naming collisions.
enum BoundParametersIndices : unsigned int {
  // Local position on the reference surface.
  // This is intentionally named different from the position components in
  // the other data vectors, to clarify that this is defined on a surface
  // while the others are defined in free space.
  eBoundLoc0 = 0,
  eBoundLoc1 = 1,
  // Direction angles
  eBoundPhi = 2,
  eBoundTheta = 3,
  // Global inverse-momentum-like parameter, i.e. q/p or 1/p
  // The naming is inconsistent for the case of neutral track parameters where
  // the value is interpreted as 1/p not as q/p. This is intentional to avoid
  // having multiple aliases for the same element and for lack of an acceptable
  // common name.
  eBoundQOverP = 4,
  eBoundTime = 5,
  // Last uninitialized value contains the total number of components
  eBoundParametersSize,
  // The following aliases without prefix exist for historical reasons
  // Generic spatial coordinates on the local surface
  eLOC_0 = eBoundLoc0,
  eLOC_1 = eBoundLoc1,
  // Spatial coordinates on a disk in polar coordinates
  eLOC_R = eLOC_0,
  eLOC_PHI = eLOC_1,
  // Spatial coordinates on a disk in Cartesian coordinates
  eLOC_X = eLOC_0,
  eLOC_Y = eLOC_1,
  // Spatial coordinates on a cylinder
  eLOC_RPHI = eLOC_0,
  eLOC_Z = eLOC_1,
  // Closest approach coordinates on a virtual perigee surface
  eLOC_D0 = eLOC_0,
  eLOC_Z0 = eLOC_1,
  // Direction angles
  ePHI = eBoundPhi,
  eTHETA = eBoundTheta,
  eQOP = eBoundQOverP,
  eT = eBoundTime,
};

/// Underlying fundamental scalar type for bound track parameters.
using BoundParametersScalar = double;

/// Components of a free track parameters vector.
///
/// To be used to access components by named indices instead of just numbers.
/// This must be a regular `enum` and not a scoped `enum class` to allow
/// implicit conversion to an integer. The enum value are thus visible directly
/// in `namespace Acts` and are prefixed to avoid naming collisions.
enum FreeParametersIndices : unsigned int {
  // Spatial position
  // The spatial position components must be stored as one continous block.
  eFreePos0 = 0u,
  eFreePos1 = eFreePos0 + 1u,
  eFreePos2 = eFreePos0 + 2u,
  // Time
  eFreeTime = 3u,
  // (Unit) direction
  // The direction components must be stored as one continous block.
  eFreeDir0 = 4u,
  eFreeDir1 = eFreeDir0 + 1u,
  eFreeDir2 = eFreeDir0 + 2u,
  // Global inverse-momentum-like parameter, i.e. q/p or 1/p
  // See BoundParametersIndices for further information
  eFreeQOverP = 7u,
  // Last uninitialized value contains the total number of components
  eFreeParametersSize,
};

/// Underlying fundamental scalar type for free track parameters.
using FreeParametersScalar = double;

}  // namespace Acts
#endif

namespace Acts {

static_assert(2 <= BoundParametersIndices::eBoundParametersSize,
              "Bound track parameters must have at least two components");

// Ensure free track parameters definition is valid.
static_assert(6 <= FreeParametersIndices::eFreeParametersSize,
              "Free track parameters must have at least six components");

// Ensure bound track parameter components/ indices are consistently defined.
static_assert(eLOC_0 != eLOC_1, "Local parameters must be differents");
static_assert(eLOC_R == eLOC_0 or eLOC_R == eLOC_1,
              "Local radius must be a local parameter");
static_assert(eLOC_PHI == eLOC_0 or eLOC_PHI == eLOC_1,
              "Local phi must be a local parameter");
static_assert(eLOC_RPHI == eLOC_0 or eLOC_RPHI == eLOC_1,
              "Local r*phi must be a local parameter");
static_assert(eLOC_Z == eLOC_0 or eLOC_Z == eLOC_1,
              "Local z must be a local parameter");
static_assert(eLOC_X == eLOC_0 or eLOC_X == eLOC_1,
              "Local x must be a local parameter");
static_assert(eLOC_Y == eLOC_0 or eLOC_Y == eLOC_1,
              "Local y must be a local parameter");
static_assert(eLOC_D0 == eLOC_0 or eLOC_D0 == eLOC_1,
              "D0 must be a local parameter");
static_assert(eLOC_Z0 == eLOC_0 or eLOC_Z0 == eLOC_1,
              "Z0 must be a local parameter");

// Ensure free track parameter components/ indices are consistently defined.
static_assert(eFreePos1 == eFreePos0 + 1u, "Position must be continous");
static_assert(eFreePos2 == eFreePos0 + 2u, "Position must be continous");
static_assert(eFreeDir1 == eFreeDir0 + 1u, "Direction must be continous");
static_assert(eFreeDir2 == eFreeDir0 + 2u, "Direction must be continous");

namespace detail {
template <BoundParametersIndices>
struct BoundParameterTraits;
template <>
struct BoundParameterTraits<BoundParametersIndices::eBoundLoc0> {
  using type = local_parameter;
};
template <>
struct BoundParameterTraits<BoundParametersIndices::eBoundLoc1> {
  using type = local_parameter;
};
template <>
struct BoundParameterTraits<BoundParametersIndices::eBoundPhi> {
  static constexpr double pMin() { return -M_PI; }
  static constexpr double pMax() { return M_PI; }
  using type = cyclic_parameter<double, pMin, pMax>;
};
template <>
struct BoundParameterTraits<BoundParametersIndices::eBoundTheta> {
  static constexpr double pMin() { return 0; }
  static constexpr double pMax() { return M_PI; }
  using type = bound_parameter<double, pMin, pMax>;
};
template <>
struct BoundParameterTraits<BoundParametersIndices::eBoundQOverP> {
  using type = unbound_parameter;
};
template <>
struct BoundParameterTraits<BoundParametersIndices::eBoundTime> {
  using type = unbound_parameter;
};

template <FreeParametersIndices>
struct FreeParameterTraits;
template <>
struct FreeParameterTraits<FreeParametersIndices::eFreePos0> {
  using type = unbound_parameter;
};
template <>
struct FreeParameterTraits<FreeParametersIndices::eFreePos1> {
  using type = unbound_parameter;
};
template <>
struct FreeParameterTraits<FreeParametersIndices::eFreePos2> {
  using type = unbound_parameter;
};
template <>
struct FreeParameterTraits<FreeParametersIndices::eFreeTime> {
  using type = unbound_parameter;
};
template <>
struct FreeParameterTraits<FreeParametersIndices::eFreeDir0> {
  using type = unbound_parameter;
};
template <>
struct FreeParameterTraits<FreeParametersIndices::eFreeDir1> {
  using type = unbound_parameter;
};
template <>
struct FreeParameterTraits<FreeParametersIndices::eFreeDir2> {
  using type = unbound_parameter;
};
template <>
struct FreeParameterTraits<FreeParametersIndices::eFreeQOverP> {
  using type = unbound_parameter;
};

template <typename parameter_indices_t>
struct ParametersSize;
template <>
struct ParametersSize<BoundParametersIndices> {
  static constexpr unsigned int size =
      static_cast<unsigned int>(BoundParametersIndices::eBoundParametersSize);
};
template <>
struct ParametersSize<FreeParametersIndices> {
  static constexpr unsigned int size =
      static_cast<unsigned int>(FreeParametersIndices::eFreeParametersSize);
};
}  // namespace detail

/// Single bound track parameter type for value constrains.
///
/// The singular name is not a typo since this describes individual components.
template <BoundParametersIndices kIndex>
using BoundParameterType = typename detail::BoundParameterTraits<kIndex>::type;

/// Single free track parameter type for value constrains.
///
/// The singular name is not a typo since this describes individual components.
template <FreeParametersIndices kIndex>
using FreeParameterType = typename detail::FreeParameterTraits<kIndex>::type;

/// Access the (Bound/Free)ParameterType through common struct
///
/// @tparam T Parameter indices enum
/// @tparam I Index of @p T
template <typename T, T I>
struct ParameterTypeFor {};

/// Access for @c BoundParametersIndices
template <BoundParametersIndices I>
struct ParameterTypeFor<BoundParametersIndices, I> {
  using type = BoundParameterType<I>;
};

/// Access for @c FreeParametersIndices
template <FreeParametersIndices I>
struct ParameterTypeFor<FreeParametersIndices, I> {
  using type = FreeParameterType<I>;
};

// The following matrix and vector types are automatically derived from the
// indices enums and scalar typedefs.

// Matrix and vector types related to bound track parameters.

using BoundVector = ActsVector<BoundParametersScalar, eBoundParametersSize>;
using BoundRowVector =
    ActsRowVector<BoundParametersScalar, eBoundParametersSize>;
using BoundMatrix = ActsMatrix<BoundParametersScalar, eBoundParametersSize,
                               eBoundParametersSize>;
using BoundSymMatrix =
    ActsSymMatrix<BoundParametersScalar, eBoundParametersSize>;

using LocalCartesianToBoundLocalMatrix =
    ActsMatrix<BoundParametersScalar, 2, 3>;

// Matrix and vector types related to free track parameters.

using FreeVector = ActsVector<FreeParametersScalar, eFreeParametersSize>;
using FreeRowVector = ActsRowVector<FreeParametersScalar, eFreeParametersSize>;
using FreeMatrix =
    ActsMatrix<FreeParametersScalar, eFreeParametersSize, eFreeParametersSize>;
using FreeSymMatrix = ActsSymMatrix<FreeParametersScalar, eFreeParametersSize>;

// Mapping to bound track parameters.
//
// Assumes that matrices represent maps from another space into the space of
// bound track parameters. Thus, the bound parameters scalar type is sufficient
// to retain accuracy.

using FreeToBoundMatrix = ActsMatrix<BoundParametersScalar,
                                     eBoundParametersSize, eFreeParametersSize>;

// Mapping to free track parameters.
//
// Assumes that matrices represent maps from another space into the space of
// free track parameters. Thus, the free parameters scalar type is sufficient
// to retain accuracy.

using BoundToFreeMatrix =
    ActsMatrix<FreeParametersScalar, eFreeParametersSize, eBoundParametersSize>;

// For backward compatibility. New code must use the more explicit
// `BoundParameters{Indices,Scalar,Traits}...` types.
using ParDef = BoundParametersIndices;
using ParID_t = BoundParametersIndices;
using ParValue_t = BoundParametersScalar;

}  // namespace Acts
