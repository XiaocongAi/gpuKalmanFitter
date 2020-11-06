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
class ISurfaceMaterial {
 public:
  /// Constructor
  ISurfaceMaterial() = default;

  /// Constructor
  ///
  /// @param splitFactor is the splitting ratio between pre/post update
  ISurfaceMaterial(double splitFactor) : m_splitFactor(splitFactor) {}

  /// Destructor
  ~ISurfaceMaterial() = default;

  /// Scale operator
  ///
  /// @param scale is the scale factor applied
  ISurfaceMaterial& operator*=(double scale);

  /// Return method for full material description of the Surface
  /// - from local coordinate on the surface
  ///
  /// @param lp is the local position used for the (eventual) lookup
  ///
  /// @return const MaterialSlab
  const MaterialSlab& materialSlab(const Vector2D& lp) const;

  /// Return method for full material description of the Surface
  /// - from the global coordinates
  ///
  /// @param gp is the global position used for the (eventual) lookup
  ///
  /// @return const MaterialSlab
  const MaterialSlab& materialSlab(const Vector3D& gp) const;

  /// Direct access via bins to the MaterialSlab
  ///
  /// @param ib0 is the material bin in dimension 0
  /// @param ib1 is the material bin in dimension 1
  const MaterialSlab& materialSlab(size_t ib0, size_t ib1) const;

  /// Update pre factor
  ///
  /// @param pDir is the navigation direction through the surface
  /// @param mStage is the material update directive (onapproach, full, onleave)
  double factor(NavigationDirection pDir, MaterialUpdateStage mStage) const;

  /// Return method for fully scaled material description of the Surface
  /// - from local coordinate on the surface
  ///
  /// @param lp is the local position used for the (eventual) lookup
  /// @param pDir is the navigation direction through the surface
  /// @param mStage is the material update directive (onapproach, full, onleave)
  ///
  /// @return MaterialSlab
  MaterialSlab materialSlab(const Vector2D& lp, NavigationDirection pDir,
                            MaterialUpdateStage mStage) const;

  /// Return method for full material description of the Surface
  /// - from the global coordinates
  ///
  /// @param gp is the global position used for the (eventual) lookup
  /// @param pDir is the navigation direction through the surface
  /// @param mStage is the material update directive (onapproach, full, onleave)
  ///
  /// @return MaterialSlab
  MaterialSlab materialSlab(const Vector3D& gp, NavigationDirection pDir,
                            MaterialUpdateStage mStage) const;

 protected:
  double m_splitFactor{1.};  //!< the split factor in favour of oppositePre
};

template <typename Derived>
inline ISurfaceMaterial& operator*=(double scale) {
 return static_cast<Derived &>(*this).operator*=(double scale);
}  

template<typename Derived>
inline const MaterialSlab& materialSlab(const Vector2D& lp) const{
 return static_cast<Derived &>(*this).materialSlab(const Vector2D& lp);
}

template<typename Derived>
inline const MaterialSlab& materialSlab(const Vector3D& gp) const {
 return static_cast<Derived &>(*this).materialSlab(const Vector3D& gp);
}

template<typename Derived>
inline const MaterialSlab& materialSlab(size_t ib0, size_t ib1) const {
 return static_cast<Derived &>(*this).materialSlab(size_t ib0, size_t ib1);
}

inline double ISurfaceMaterial::factor(NavigationDirection pDir,
                                       MaterialUpdateStage mStage) const {
  if (mStage == Acts::fullUpdate) {
    return 1.;
  }
  return (pDir * mStage > 0 ? m_splitFactor : 1. - m_splitFactor);
}

inline MaterialSlab ISurfaceMaterial::materialSlab(
    const Vector2D& lp, NavigationDirection pDir,
    MaterialUpdateStage mStage) const {
  // The plain material properties associated to this bin
  MaterialSlab plainMatProp = materialSlab(lp);
  // Scale if you have material to scale
  if (plainMatProp) {
    double scaleFactor = factor(pDir, mStage);
    if (scaleFactor == 0.) {
      return MaterialSlab();
    }
    plainMatProp.scaleThickness(scaleFactor);
  }
  return plainMatProp;
}

inline MaterialSlab ISurfaceMaterial::materialSlab(
    const Vector3D& gp, NavigationDirection pDir,
    MaterialUpdateStage mStage) const {
  // The plain material properties associated to this bin
  MaterialSlab plainMatProp = materialSlab(gp);
  // Scale if you have material to scale
  if (plainMatProp) {
    double scaleFactor = factor(pDir, mStage);
    if (scaleFactor == 0.) {
      return MaterialSlab();
    }
    plainMatProp.scaleThickness(scaleFactor);
  }
  return plainMatProp;
}

}  // namespace Acts
