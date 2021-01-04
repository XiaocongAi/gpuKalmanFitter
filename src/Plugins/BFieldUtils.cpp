// This file is part of the Acts project.
//
// Copyright (C) 2017 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Plugins/BFieldUtils.hpp"
#include "MagneticField/BFieldMapUtils.hpp"
#include "Utilities/detail/Axis.hpp"
#include "Utilities/detail/Grid.hpp"
#include <fstream>

Acts::InterpolatedBFieldMapper<
    Acts::detail::Grid<Acts::Vector2D, Acts::detail::EquidistantAxis,
                       Acts::detail::EquidistantAxis>>
BField::txt::fieldMapperRZ(std::function<size_t(std::array<size_t, 2> binsRZ,
                                                std::array<size_t, 2> nBinsRZ)>
                               localToGlobalBin,
                           std::string fieldMapFile, ActsScalar lengthUnit,
                           ActsScalar BFieldUnit, size_t nPoints,
                           bool firstQuadrant) {
  /// [1] Read in field map file
  // Grid position points in r and z
  std::vector<ActsScalar> rPos;
  std::vector<ActsScalar> zPos;
  // components of magnetic field on grid points
  std::vector<Acts::Vector2D> bField;
  // reserve estimated size
  rPos.reserve(nPoints);
  zPos.reserve(nPoints);
  bField.reserve(nPoints);
  // [1] Read in file and fill values
  std::ifstream map_file(fieldMapFile.c_str(), std::ios::in);
  std::string line;
  ActsScalar r = 0., z = 0.;
  ActsScalar br = 0., bz = 0.;
  while (std::getline(map_file, line)) {
    if (line.empty() || line[0] == '%' || line[0] == '#' ||
        line.find_first_not_of(' ') == std::string::npos)
      continue;

    std::istringstream tmp(line);
    tmp >> r >> z >> br >> bz;
    rPos.push_back(r);
    zPos.push_back(z);
    bField.push_back(Acts::Vector2D(br, bz));
  }
  map_file.close();
  /// [2] use helper function in core
  return Acts::fieldMapperRZ(localToGlobalBin, rPos, zPos, bField, lengthUnit,
                             BFieldUnit, firstQuadrant);
}

Acts::InterpolatedBFieldMapper<Acts::detail::Grid<
    Acts::Vector3D, Acts::detail::EquidistantAxis,
    Acts::detail::EquidistantAxis, Acts::detail::EquidistantAxis>>
BField::txt::fieldMapperXYZ(
    std::function<size_t(std::array<size_t, 3> binsXYZ,
                         std::array<size_t, 3> nBinsXYZ)>
        localToGlobalBin,
    std::string fieldMapFile, ActsScalar lengthUnit, ActsScalar BFieldUnit,
    size_t nPoints, bool firstOctant) {
  /// [1] Read in field map file
  // Grid position points in x, y and z
  std::vector<ActsScalar> xPos;
  std::vector<ActsScalar> yPos;
  std::vector<ActsScalar> zPos;
  // components of magnetic field on grid points
  std::vector<Acts::Vector3D> bField;
  // reserve estimated size
  xPos.reserve(nPoints);
  yPos.reserve(nPoints);
  zPos.reserve(nPoints);
  bField.reserve(nPoints);

  // [1] Read in file and fill values
  std::ifstream map_file(fieldMapFile.c_str(), std::ios::in);
  std::string line;
  ActsScalar x = 0., y = 0., z = 0.;
  ActsScalar bx = 0., by = 0., bz = 0.;
  while (std::getline(map_file, line)) {
    if (line.empty() || line[0] == '%' || line[0] == '#' ||
        line.find_first_not_of(' ') == std::string::npos)
      continue;

    std::istringstream tmp(line);
    tmp >> x >> y >> z >> bx >> by >> bz;
    xPos.push_back(x);
    yPos.push_back(y);
    zPos.push_back(z);
    bField.push_back(Acts::Vector3D(bx, by, bz));
  }
  map_file.close();

  return Acts::fieldMapperXYZ(localToGlobalBin, xPos, yPos, zPos, bField,
                              lengthUnit, BFieldUnit, firstOctant);
}
