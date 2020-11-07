// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Material/ISurfaceMaterial.hpp"
#include "Material/MaterialSlab.hpp"
#include "Utilities/Definitions.hpp"

#include <iosfwd>

namespace Acts {

/// @class HomogeneousSurfaceMaterial
///
/// It extends the ISurfaceMaterial virutal base class to describe
/// a simple homogeneous material on a surface
class HomogeneousSurfaceMaterial : public ISurfaceMaterial<HomogeneousSurfaceMaterial> {
 public:
  /// Default Constructor - defaulted
  HomogeneousSurfaceMaterial() = default;

  /// Explicit constructor
  ///
  /// @param full are the full material properties
  /// @param splitFactor is the split for pre/post update
  ACTS_DEVICE_FUNC HomogeneousSurfaceMaterial(const MaterialSlab& full, double splitFactor = 1.);

  /// Copy Constructor
  ///
  /// @param hsm is the source material
  HomogeneousSurfaceMaterial(const HomogeneousSurfaceMaterial& hsm) = default;

  /// Copy Move Constructor
  ///
  /// @param hsm is the source material
  HomogeneousSurfaceMaterial(HomogeneousSurfaceMaterial&& hsm) = default;

  /// Destructor
  ~HomogeneousSurfaceMaterial() = default;

  /// Assignment operator
  ///
  /// @param hsm is the source material
  HomogeneousSurfaceMaterial& operator=(const HomogeneousSurfaceMaterial& hsm) =
      default;

  /// Assignment Move operator
  ///
  /// @param hsm is the source material
  HomogeneousSurfaceMaterial& operator=(HomogeneousSurfaceMaterial&& hsm) =
      default;

  /// Scale operator
  /// - it is effectively a thickness scaling
  ///
  /// @param scale is the scale factor
  ACTS_DEVICE_FUNC HomogeneousSurfaceMaterial& operator*=(double scale);

  /// Equality operator
  ///
  /// @param hsm is the source material
  ACTS_DEVICE_FUNC bool operator==(const HomogeneousSurfaceMaterial& hsm) const;

  /// Check if the material is valid, i.e. it is finite and not vacuum.
   ACTS_DEVICE_FUNC constexpr operator bool() const {
     return bool(m_fullMaterial);
  }

  /// @copydoc SurfaceMaterial::materialSlab(const Vector2D&)
  ///
  /// @note the input parameter is ignored
  ACTS_DEVICE_FUNC const MaterialSlab& materialSlab(const Vector2D& lp) const;

  /// @copydoc SurfaceMaterial::materialSlab(const Vector3D&)
  ///
  /// @note the input parameter is ignored
  ACTS_DEVICE_FUNC const MaterialSlab& materialSlab(const Vector3D& gp) const;

  /// @copydoc SurfaceMaterial::materialSlab(size_t, size_t)
  ///
  /// @param ib0 The bin at local 0 for retrieving the material
  /// @param ib1 The bin at local 1 for retrieving the material
  ///
  /// @note the input parameter is ignored
  ACTS_DEVICE_FUNC const MaterialSlab& materialSlab(size_t ib0, size_t ib1) const;

  /// The inherited methods - for MaterialSlab access
  using ISurfaceMaterial::materialSlab;

  /// The interited methods - for scale access
  using ISurfaceMaterial::factor;

 private:
  /// The five different MaterialSlab
  MaterialSlab m_fullMaterial = MaterialSlab();
};

inline const MaterialSlab& HomogeneousSurfaceMaterial::materialSlab(
    const Vector2D& /*lp*/) const {
  return (m_fullMaterial);
}

inline const MaterialSlab& HomogeneousSurfaceMaterial::materialSlab(
    const Vector3D& /*gp*/) const {
  return (m_fullMaterial);
}

inline const MaterialSlab& HomogeneousSurfaceMaterial::materialSlab(
    size_t /*ib0*/, size_t /*ib1*/) const {
  return (m_fullMaterial);
}

inline bool HomogeneousSurfaceMaterial::operator==(
    const HomogeneousSurfaceMaterial& hsm) const {
  return (m_fullMaterial == hsm.m_fullMaterial);
}

}  // namespace Acts
