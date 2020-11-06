// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Material/MaterialSlab.hpp"

#include <limits>
#include <ostream>

static constexpr auto eps = 2 * std::numeric_limits<float>::epsilon();

Acts::MaterialSlab::MaterialSlab(float thickness) : m_thickness(thickness) {}

Acts::MaterialSlab::MaterialSlab(const Material& material, float thickness)
    : m_material(material),
      m_thickness(thickness),
      m_thicknessInX0((eps < material.X0()) ? (thickness / material.X0()) : 0),
      m_thicknessInL0((eps < material.L0()) ? (thickness / material.L0()) : 0) {
}

void Acts::MaterialSlab::scaleThickness(float scale) {
  m_thickness *= scale;
  m_thicknessInX0 *= scale;
  m_thicknessInL0 *= scale;
}
