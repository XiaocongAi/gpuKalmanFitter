// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "MagneticField/SolenoidBField.hpp"
//#include "Utilities/Helpers.hpp"

#include "Math/Math.hpp"

Acts::SolenoidBField::SolenoidBField(Config config) : m_cfg(std::move(config)) {
  m_dz = m_cfg.length / m_cfg.nCoils;
  m_R2 = m_cfg.radius * m_cfg.radius;
  // we need to scale so we reproduce the expected B field strength
  // at the center of the solenoid
  Vector2D field = multiCoilField({0, 0}, 1.); // scale = 1
  m_scale = m_cfg.bMagCenter / field.norm();
}

Acts::Vector3D Acts::SolenoidBField::getField(const Vector3D &position) const {
  using VectorHelpers::perp;
  Vector2D rzPos(perp(position), position.z());
  Vector2D rzField = multiCoilField(rzPos, m_scale);
  Vector3D xyzField(0, 0, rzField[1]);

  if (rzPos[0] != 0.) {
    // add xy field component, radially symmetric
    Vector3D rDir = Vector3D(position.x(), position.y(), 0).normalized();
    xyzField += rDir * rzField[0];
  }

  return xyzField;
}

Acts::Vector3D Acts::SolenoidBField::getField(const Vector3D &position,
                                              Cache & /*cache*/) const {
  return getField(position);
}

Acts::Vector2D Acts::SolenoidBField::getField(const Vector2D &position) const {
  return multiCoilField(position, m_scale);
}

Acts::Vector3D Acts::SolenoidBField::getFieldGradient(
    const Vector3D &position, ActsMatrixD<3, 3> & /*derivative*/) const {
  return getField(position);
}

Acts::Vector3D
Acts::SolenoidBField::getFieldGradient(const Vector3D &position,
                                       ActsMatrixD<3, 3> & /*derivative*/,
                                       Cache & /*cache*/) const {
  return getField(position);
}

Acts::Vector2D Acts::SolenoidBField::multiCoilField(const Vector2D &pos,
                                                    ActsScalar scale) const {
  // iterate over all coils
  Vector2D resultField(0, 0);
  for (size_t coil = 0; coil < m_cfg.nCoils; coil++) {
    Vector2D shiftedPos =
        Vector2D(pos[0], pos[1] + m_cfg.length * 0.5 - m_dz * (coil + 0.5));
    resultField += singleCoilField(shiftedPos, scale);
  }

  return resultField;
}

Acts::Vector2D Acts::SolenoidBField::singleCoilField(const Vector2D &pos,
                                                     ActsScalar scale) const {
  return {B_r(pos, scale), B_z(pos, scale)};
}

ActsScalar Acts::SolenoidBField::B_r(const Vector2D &pos,
                                     ActsScalar scale) const {
  //              _
  //     2       /  pi / 2          2    2          - 1 / 2
  // E (k )  =   |         ( 1  -  k  sin {theta} )         dtheta
  //  1         _/  0
  //              _          ____________________
  //     2       /  pi / 2| /       2    2
  // E (k )  =   |        |/ 1  -  k  sin {theta} dtheta
  //  2         _/  0

  ActsScalar r = std::abs(pos[0]);
  ActsScalar z = pos[1];

  if (r == 0) {
    return 0.;
  }

  //                            _                             _
  //              mu  I        |  /     2 \                    |
  //                0     kz   |  |2 - k  |    2          2    |
  // B (r, z)  =  ----- ------ |  |-------|E (k )  -  E (k )   |
  //  r            4pi     ___ |  |      2| 2          1       |
  //                    | /  3 |_ \2 - 2k /                   _|
  //                    |/ Rr
  ActsScalar k_2 = k2(r, z);
  ActsScalar k = std::sqrt(k_2);
  ActsScalar constant =
      scale * k * z / (4 * M_PI * std::sqrt(m_cfg.radius * r * r * r));

  ActsScalar B = (2. - k_2) / (2. - 2. * k_2) * Math::comp_ellint_2(k_2) -
                 Math::comp_ellint_1(k_2);

  // pos[0] is still signed!
  return r / pos[0] * constant * B;
}

ActsScalar Acts::SolenoidBField::B_z(const Vector2D &pos,
                                     ActsScalar scale) const {
  //              _
  //     2       /  pi / 2          2    2          - 1 / 2
  // E (k )  =   |         ( 1  -  k  sin {theta} )         dtheta
  //  1         _/  0
  //              _          ____________________
  //     2       /  pi / 2| /       2    2
  // E (k )  =   |        |/ 1  -  k  sin {theta} dtheta
  //  2         _/  0

  ActsScalar r = std::abs(pos[0]);
  ActsScalar z = pos[1];

  //                         _                                       _
  //             mu  I      |  /         2      \                     |
  //               0     k  |  | (R + r)k  - 2r |     2          2    |
  // B (r,z)  =  ----- ---- |  | -------------- | E (k )  +  E (k )   |
  //  z           4pi    __ |  |           2    |  2          1       |
  //                   |/Rr |_ \   2r(1 - k )   /                    _|

  if (r == 0) {
    ActsScalar res =
        scale / 2. * m_R2 / (std::sqrt(m_R2 + z * z) * (m_R2 + z * z));
    return res;
  }

  ActsScalar k_2 = k2(r, z);
  ActsScalar k = std::sqrt(k_2);
  ActsScalar constant = scale * k / (4 * M_PI * std::sqrt(m_cfg.radius * r));
  ActsScalar B = ((m_cfg.radius + r) * k_2 - 2. * r) / (2. * r * (1. - k_2)) *
                     Math::comp_ellint_2(k_2) +
                 Math::comp_ellint_1(k_2);

  return constant * B;
}

ActsScalar Acts::SolenoidBField::k2(ActsScalar r, ActsScalar z) const {
  //  2           4Rr
  // k   =  ---------------
  //               2      2
  //        (R + r)   +  z
  return 4 * m_cfg.radius * r /
         ((m_cfg.radius + r) * (m_cfg.radius + r) + z * z);
}
