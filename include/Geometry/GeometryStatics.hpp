// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// GeometryStatics.h, Acts project
///////////////////////////////////////////////////////////////////

#pragma once
#include "Utilities/Definitions.hpp"

/// Define statics for Geometry in Tracking
///
namespace Acts {

// transformations

static const Transform3D s_idTransform =
    Transform3D::Identity(); //!< idendity transformation
static const Rotation3D s_idRotation =
    Rotation3D::Identity(); //!< idendity rotation

// axis system
static const Vector3D s_xAxis(1, 0, 0); //!< global x Axis;
static const Vector3D s_yAxis(0, 1, 0); //!< global y Axis;
static const Vector3D s_zAxis(0, 0, 1); //!< global z Axis;

// unit vectors
static const Vector2D s_origin2D(0., 0.);

// origin

static const Vector3D s_origin(0, 0, 0); //!< origin position

namespace detail {
static const double _helper[9] = {0., 1., 0., 1., 0., 0., 0., 0., -1.};
}

static const RotationMatrix3D s_idRotationZinverse(detail::_helper);

static const uint64_t volume_mask = 0xff00000000000000; // 255 volumes (16^2-1)
static const uint64_t boundary_mask =
    0x00ff000000000000; // 255 boundaries (16^2-1)
static const uint64_t layer_mask = 0x0000fff000000000; // 4095 layers (16^3-1)
static const uint64_t approach_mask =
    0x0000000ff0000000; // 255 approach surfaces (16^2-1)
static const uint64_t sensitive_mask =
    0x000000000fffffff; // (2^28)-1 sensitive surfaces (16^7-1)

} // namespace Acts
