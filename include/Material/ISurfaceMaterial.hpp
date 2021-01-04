// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once
#include "Material/MaterialSlab.hpp"
#include "Utilities/Definitions.hpp"

#include <memory>
#include <vector>

namespace Acts {

/// @class ISurfaceMaterial
///
/// Virtual base class of surface based material description
///
/// MaterialSlab that are associated to a surface,
/// extended by certain special representations (binned, homogenous)
///
template <typename Derived> class ISurfaceMaterial {
public:
  /// Constructor
  ISurfaceMaterial() = default;

  /// Constructor
  ///
  /// @param splitFactor is the splitting ratio between pre/post update
  ACTS_DEVICE_FUNC ISurfaceMaterial(ActsScalar splitFactor)
      : m_splitFactor(splitFactor) {}

  /// Destructor
  ~ISurfaceMaterial() = default;

  /// Scale operator
  ///
  /// @param scale is the scale factor applied
  ACTS_DEVICE_FUNC ISurfaceMaterial &operator*=(ActsScalar scale);

  /// Return method for full material description of the Surface
  /// - from local coordinate on the surface
  ///
  /// @param lp is the local position used for the (eventual) lookup
  ///
  /// @return const MaterialSlab
  ACTS_DEVICE_FUNC const MaterialSlab &materialSlab(const Vector2D &lp) const;

  /// Return method for full material description of the Surface
  /// - from the global coordinates
  ///
  /// @param gp is the global position used for the (eventual) lookup
  ///
  /// @return const MaterialSlab
  ACTS_DEVICE_FUNC const MaterialSlab &materialSlab(const Vector3D &gp) const;

  /// Direct access via bins to the MaterialSlab
  ///
  /// @param ib0 is the material bin in dimension 0
  /// @param ib1 is the material bin in dimension 1
  ACTS_DEVICE_FUNC const MaterialSlab &materialSlab(size_t ib0,
                                                    size_t ib1) const;

  /// Update pre factor
  ///
  /// @param pDir is the navigation direction through the surface
  /// @param mStage is the material update directive (onapproach, full, onleave)
  ACTS_DEVICE_FUNC ActsScalar factor(NavigationDirection pDir,
                                     MaterialUpdateStage mStage) const;

  /// Return method for fully scaled material description of the Surface
  /// - from local coordinate on the surface
  ///
  /// @param lp is the local position used for the (eventual) lookup
  /// @param pDir is the navigation direction through the surface
  /// @param mStage is the material update directive (onapproach, full, onleave)
  ///
  /// @return MaterialSlab
  ACTS_DEVICE_FUNC MaterialSlab materialSlab(const Vector2D &lp,
                                             NavigationDirection pDir,
                                             MaterialUpdateStage mStage) const;

  /// Return method for full material description of the Surface
  /// - from the global coordinates
  ///
  /// @param gp is the global position used for the (eventual) lookup
  /// @param pDir is the navigation direction through the surface
  /// @param mStage is the material update directive (onapproach, full, onleave)
  ///
  /// @return MaterialSlab
  ACTS_DEVICE_FUNC MaterialSlab materialSlab(const Vector3D &gp,
                                             NavigationDirection pDir,
                                             MaterialUpdateStage mStage) const;

protected:
  ActsScalar m_splitFactor{1.}; //!< the split factor in favour of oppositePre
};

template <typename Derived>
inline ISurfaceMaterial<Derived> &
ISurfaceMaterial<Derived>::operator*=(ActsScalar scale) {
  return static_cast<Derived &>(*this).operator*=(scale);
}

template <typename Derived>
inline const MaterialSlab &
ISurfaceMaterial<Derived>::materialSlab(const Vector2D &lp) const {
  return static_cast<const Derived &>(*this).materialSlab(lp);
}

template <typename Derived>
inline const MaterialSlab &
ISurfaceMaterial<Derived>::materialSlab(const Vector3D &gp) const {
  return static_cast<const Derived &>(*this).materialSlab(gp);
}

template <typename Derived>
inline const MaterialSlab &
ISurfaceMaterial<Derived>::materialSlab(size_t ib0, size_t ib1) const {
  return static_cast<const Derived &>(*this).materialSlab(ib0, ib1);
}

template <typename Derived>
inline ActsScalar
ISurfaceMaterial<Derived>::factor(NavigationDirection pDir,
                                  MaterialUpdateStage mStage) const {
  if (mStage == Acts::fullUpdate) {
    return 1.;
  }
  return (pDir * mStage > 0 ? m_splitFactor : 1. - m_splitFactor);
}

template <typename Derived>
inline MaterialSlab
ISurfaceMaterial<Derived>::materialSlab(const Vector2D &lp,
                                        NavigationDirection pDir,
                                        MaterialUpdateStage mStage) const {
  // The plain material properties associated to this bin
  MaterialSlab plainMatProp = materialSlab(lp);
  // Scale if you have material to scale
  if (plainMatProp) {
    ActsScalar scaleFactor = factor(pDir, mStage);
    if (scaleFactor == 0.) {
      return MaterialSlab();
    }
    plainMatProp.scaleThickness(scaleFactor);
  }
  return plainMatProp;
}

template <typename Derived>
inline MaterialSlab
ISurfaceMaterial<Derived>::materialSlab(const Vector3D &gp,
                                        NavigationDirection pDir,
                                        MaterialUpdateStage mStage) const {
  // The plain material properties associated to this bin
  MaterialSlab plainMatProp = materialSlab(gp);
  // Scale if you have material to scale
  if (plainMatProp) {
    ActsScalar scaleFactor = factor(pDir, mStage);
    if (scaleFactor == 0.) {
      return MaterialSlab();
    }
    plainMatProp.scaleThickness(scaleFactor);
  }
  return plainMatProp;
}

} // namespace Acts
