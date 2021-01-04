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
static const ActsScalar _helper[9] = {0., 1., 0., 1., 0., 0., 0., 0., -1.};
}

static const RotationMatrix3D s_idRotationZinverse(detail::_helper);

static const uint32_t volume_mask = 0xf0000000;   // 15 volumes (16-1)
static const uint32_t boundary_mask = 0x0f000000; // 15 boundaries (16-1)
static const uint32_t layer_mask = 0x00ff0000;    // 255 layers (16^2-1)
static const uint32_t approach_mask = 0x0000f000; // 15 approach surfaces (16-1)
static const uint32_t sensitive_mask =
    0x00000fff; // (2^12)-1 sensitive surfaces (16^3-1)

} // namespace Acts
