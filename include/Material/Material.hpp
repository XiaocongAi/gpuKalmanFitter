// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Utilities/Definitions.hpp"

#include <iosfwd>
#include <limits>

namespace Acts {

/// Material description for interactions with matter.
///
/// The following parameters are used to specify the material and its
/// interactions with traversing particles:
///
/// - radiation length X0 (native length units)
/// - nuclear interaction length L0 (native length units)
/// - relative atomic mass Ar (unitless number)
/// - nuclear charge number Z (elementary charge e)
/// - molar density (native amount-of-substance unit / (native length unit)³)
///
/// The parameters can be effective or average parameters e.g. when a mixture
/// of materials is described.
///
/// @note Always use the opaque parameters vector to serialize/deserialize the
///   material information. Since the internal storage might be different from
///   the external accessors, this ensures that always the numerically optimal
///   parameters are stored. Use the `ParametersVector` type and do not assume
///   any particular size since we might consider to store more parameters in
///   the future.
class Material {
public:
  using ParametersVector = Acts::ActsVectorF<5>;

  // Both mass and molar density are stored as a ActsScalar and can thus not be
  // distinguished by their types. Just changing the last element in the
  // previously existing constructor that took five ActsScalars as input to
  // represent molar density instead of mass density could have lead to
  // significant confusion compared to the previous behaviour. To avoid any
  // ambiguity, construction from separate material parameters must happen
  // through the following named constructors.

  /// Construct from material parameters using the molar density.
  ///
  /// @param X0 is the radiation length
  /// @param L0 is the nuclear interaction length
  /// @param Ar is the relative atomic mass
  /// @param Z is the nuclear charge number
  /// @param molarRho is the molar density
  ACTS_DEVICE_FUNC static Material fromMolarDensity(ActsScalar x0,
                                                    ActsScalar l0,
                                                    ActsScalar ar, ActsScalar z,
                                                    ActsScalar molarRho);
  /// Construct from material parameters using the mass density.
  ///
  /// @param X0 is the radiation length
  /// @param L0 is the nuclear interaction length
  /// @param Ar is the relative atomic mass
  /// @param Z is the nuclear charge number
  /// @param massRho is the mass density
  ///
  /// @warning Due to the choice of native mass units, using the mass density
  ///   can lead to numerical problems. Typical mass densities lead to
  ///   computations with values differing by 20+ orders of magnitude.
  ACTS_DEVICE_FUNC static Material fromMassDensity(ActsScalar x0, ActsScalar l0,
                                                   ActsScalar ar, ActsScalar z,
                                                   ActsScalar massRho);
  /// Construct a vacuum representation.
  Material() = default;
  /// Construct from an encoded parameters vector.
  ACTS_DEVICE_FUNC Material(const ParametersVector &parameters);

  Material(Material &&mat) = default;
  Material(const Material &mat) = default;
  ~Material() = default;
  Material &operator=(Material &&mat) = default;
  Material &operator=(const Material &mat) = default;

  /// Check if the material is valid, i.e. it is not vacuum.
  ACTS_DEVICE_FUNC constexpr operator bool() const { return 0.0f < m_ar; }

  /// Return the radition length. Infinity in case of vacuum.
  ACTS_DEVICE_FUNC constexpr ActsScalar X0() const { return m_x0; }
  /// Return the nuclear interaction length. Infinity in case of vacuum.
  ACTS_DEVICE_FUNC constexpr ActsScalar L0() const { return m_l0; }
  /// Return the relative atomic mass.
  ACTS_DEVICE_FUNC constexpr ActsScalar Ar() const { return m_ar; }
  /// Return the nuclear charge number.
  ACTS_DEVICE_FUNC constexpr ActsScalar Z() const { return m_z; }
  /// Return the molar density.
  ACTS_DEVICE_FUNC constexpr ActsScalar molarDensity() const {
    return m_molarRho;
  }
  /// Return the molar electron density.
  ACTS_DEVICE_FUNC constexpr ActsScalar molarElectronDensity() const {
    return m_z * m_molarRho;
  }
  /// Return the mass density.
  ACTS_DEVICE_FUNC ActsScalar massDensity() const;
  /// Return the mean electron excitation energy.
  ACTS_DEVICE_FUNC ActsScalar meanExcitationEnergy() const;

  /// Encode the properties into an opaque parameters vector.
  ACTS_DEVICE_FUNC ParametersVector parameters() const;

private:
  ActsScalar m_x0 = std::numeric_limits<ActsScalar>::infinity();
  ActsScalar m_l0 = std::numeric_limits<ActsScalar>::infinity();
  ActsScalar m_ar = 0.0f;
  ActsScalar m_z = 0.0f;
  ActsScalar m_molarRho = 0.0f;

  friend constexpr bool operator==(const Material &lhs, const Material &rhs) {
    return (lhs.m_x0 == rhs.m_x0) and (lhs.m_l0 == rhs.m_l0) and
           (lhs.m_ar == rhs.m_ar) and (lhs.m_z == rhs.m_z) and
           (lhs.m_molarRho == rhs.m_molarRho);
  }
  friend constexpr bool operator!=(const Material &lhs, const Material &rhs) {
    return !(lhs == rhs);
  }
};

} // namespace Acts

#include "Material/detail/Material.ipp"
