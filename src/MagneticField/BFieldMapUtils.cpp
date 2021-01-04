// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "MagneticField/BFieldMapUtils.hpp"
#include "MagneticField/SolenoidBField.hpp"
#include "Utilities/Helpers.hpp"
#include "Utilities/detail/Axis.hpp"
#include "Utilities/detail/Grid.hpp"
#include <iostream>

using Acts::VectorHelpers::perp;

Acts::InterpolatedBFieldMapper<
    Acts::detail::Grid<Acts::Vector2D, Acts::detail::EquidistantAxis,
                       Acts::detail::EquidistantAxis>>
Acts::fieldMapperRZ(const std::function<size_t(std::array<size_t, 2> binsRZ,
                                               std::array<size_t, 2> nBinsRZ)>
                        &localToGlobalBin,
                    std::vector<ActsScalar> rPos, std::vector<ActsScalar> zPos,
                    std::vector<Acts::Vector2D> bField, ActsScalar lengthUnit,
                    ActsScalar BFieldUnit, bool firstQuadrant) {
  // [1] Create Grid
  // sort the values
  std::sort(rPos.begin(), rPos.end());
  std::sort(zPos.begin(), zPos.end());
  // Get unique values
  rPos.erase(std::unique(rPos.begin(), rPos.end()), rPos.end());
  zPos.erase(std::unique(zPos.begin(), zPos.end()), zPos.end());
  rPos.shrink_to_fit();
  zPos.shrink_to_fit();
  // get the number of bins
  size_t nBinsR = rPos.size();
  size_t nBinsZ = zPos.size();

  // get the minimum and maximum
  auto minMaxR = std::minmax_element(rPos.begin(), rPos.end());
  auto minMaxZ = std::minmax_element(zPos.begin(), zPos.end());
  ActsScalar rMin = *minMaxR.first;
  ActsScalar zMin = *minMaxZ.first;
  ActsScalar rMax = *minMaxR.second;
  ActsScalar zMax = *minMaxZ.second;
  // calculate maxima (add one last bin, because bin value always corresponds to
  // left boundary)
  ActsScalar stepZ = std::fabs(zMax - zMin) / (nBinsZ - 1);
  ActsScalar stepR = std::fabs(rMax - rMin) / (nBinsR - 1);
  rMax += stepR;
  zMax += stepZ;
  if (firstQuadrant) {
    zMin = -(*minMaxZ.second);
    nBinsZ = 2. * nBinsZ - 1;
  }

  // Create the axis for the grid
  Acts::detail::EquidistantAxis rAxis(rMin * lengthUnit, rMax * lengthUnit,
                                      nBinsR);
  Acts::detail::EquidistantAxis zAxis(zMin * lengthUnit, zMax * lengthUnit,
                                      nBinsZ);

  // Create the grid
  using Grid_t =
      Acts::detail::Grid<Acts::Vector2D, Acts::detail::EquidistantAxis,
                         Acts::detail::EquidistantAxis>;
  Grid_t grid(std::make_tuple(std::move(rAxis), std::move(zAxis)));

  // [2] Set the bField values
  for (size_t i = 1; i <= nBinsR; ++i) {
    for (size_t j = 1; j <= nBinsZ; ++j) {
      std::array<size_t, 2> nIndices = {{rPos.size(), zPos.size()}};
      Grid_t::index_t indices = {i, j};
      if (firstQuadrant) {
        // std::vectors begin with 0 and we do not want the user needing to
        // take underflow or overflow bins in account this is why we need to
        // subtract by one
        size_t n = std::abs(int(j) - int(zPos.size()));

        grid.atLocalBins(indices) =
            bField.at(localToGlobalBin({{i - 1, n}}, nIndices)) * BFieldUnit;
      } else {
        // std::vectors begin with 0 and we do not want the user needing to
        // take underflow or overflow bins in account this is why we need to
        // subtract by one
        grid.atLocalBins(indices) =
            bField.at(localToGlobalBin({{i - 1, j - 1}}, nIndices)) *
            BFieldUnit;
      }
    }
  }
  grid.setExteriorBins(Acts::Vector2D::Zero());

  // [3] Create the transformation for the position
  // map (x,y,z) -> (r,z)
  auto transformPos = [](const Acts::Vector3D &pos) {
    return Acts::Vector2D(perp(pos), pos.z());
  };

  // [4] Create the transformation for the bfield
  // map (Br,Bz) -> (Bx,By,Bz)
  auto transformBField = [](const Acts::Vector2D &field,
                            const Acts::Vector3D &pos) {
    ActsScalar r_sin_theta_2 = pos.x() * pos.x() + pos.y() * pos.y();
    ActsScalar cos_phi, sin_phi;
    if (r_sin_theta_2 > std::numeric_limits<ActsScalar>::min()) {
      ActsScalar inv_r_sin_theta = 1. / sqrt(r_sin_theta_2);
      cos_phi = pos.x() * inv_r_sin_theta;
      sin_phi = pos.y() * inv_r_sin_theta;
    } else {
      cos_phi = 1.;
      sin_phi = 0.;
    }
    return Acts::Vector3D(field.x() * cos_phi, field.x() * sin_phi, field.y());
  };

  // [5] Create the mapper & BField Service
  // create field mapping
  return Acts::InterpolatedBFieldMapper<Grid_t>(
      Transform2DPos(), Transform2DBField(), std::move(grid));
}

Acts::InterpolatedBFieldMapper<Acts::detail::Grid<
    Acts::Vector3D, Acts::detail::EquidistantAxis,
    Acts::detail::EquidistantAxis, Acts::detail::EquidistantAxis>>
Acts::fieldMapperXYZ(const std::function<size_t(std::array<size_t, 3> binsXYZ,
                                                std::array<size_t, 3> nBinsXYZ)>
                         &localToGlobalBin,
                     std::vector<ActsScalar> xPos, std::vector<ActsScalar> yPos,
                     std::vector<ActsScalar> zPos,
                     std::vector<Acts::Vector3D> bField, ActsScalar lengthUnit,
                     ActsScalar BFieldUnit, bool firstOctant) {
  // [1] Create Grid
  // Sort the values
  std::sort(xPos.begin(), xPos.end());
  std::sort(yPos.begin(), yPos.end());
  std::sort(zPos.begin(), zPos.end());
  // Get unique values
  xPos.erase(std::unique(xPos.begin(), xPos.end()), xPos.end());
  yPos.erase(std::unique(yPos.begin(), yPos.end()), yPos.end());
  zPos.erase(std::unique(zPos.begin(), zPos.end()), zPos.end());
  xPos.shrink_to_fit();
  yPos.shrink_to_fit();
  zPos.shrink_to_fit();
  // get the number of bins
  size_t nBinsX = xPos.size();
  size_t nBinsY = yPos.size();
  size_t nBinsZ = zPos.size();

  // get the minimum and maximum
  auto minMaxX = std::minmax_element(xPos.begin(), xPos.end());
  auto minMaxY = std::minmax_element(yPos.begin(), yPos.end());
  auto minMaxZ = std::minmax_element(zPos.begin(), zPos.end());
  // Create the axis for the grid
  // get minima
  ActsScalar xMin = *minMaxX.first;
  ActsScalar yMin = *minMaxY.first;
  ActsScalar zMin = *minMaxZ.first;
  // get maxima
  ActsScalar xMax = *minMaxX.second;
  ActsScalar yMax = *minMaxY.second;
  ActsScalar zMax = *minMaxZ.second;
  // calculate maxima (add one last bin, because bin value always corresponds to
  // left boundary)
  ActsScalar stepZ = std::fabs(zMax - zMin) / (nBinsZ - 1);
  ActsScalar stepY = std::fabs(yMax - yMin) / (nBinsY - 1);
  ActsScalar stepX = std::fabs(xMax - xMin) / (nBinsX - 1);
  xMax += stepX;
  yMax += stepY;
  zMax += stepZ;

  // If only the first octant is given
  if (firstOctant) {
    xMin = -*minMaxX.second;
    yMin = -*minMaxY.second;
    zMin = -*minMaxZ.second;
    nBinsX = 2 * nBinsX - 1;
    nBinsY = 2 * nBinsY - 1;
    nBinsZ = 2 * nBinsZ - 1;
  }
  Acts::detail::EquidistantAxis xAxis(xMin * lengthUnit, xMax * lengthUnit,
                                      nBinsX);
  Acts::detail::EquidistantAxis yAxis(yMin * lengthUnit, yMax * lengthUnit,
                                      nBinsY);
  Acts::detail::EquidistantAxis zAxis(zMin * lengthUnit, zMax * lengthUnit,
                                      nBinsZ);
  // Create the grid
  using Grid_t =
      Acts::detail::Grid<Acts::Vector3D, Acts::detail::EquidistantAxis,
                         Acts::detail::EquidistantAxis,
                         Acts::detail::EquidistantAxis>;
  Grid_t grid(
      std::make_tuple(std::move(xAxis), std::move(yAxis), std::move(zAxis)));

  // [2] Set the bField values
  for (size_t i = 1; i <= nBinsX; ++i) {
    for (size_t j = 1; j <= nBinsY; ++j) {
      for (size_t k = 1; k <= nBinsZ; ++k) {
        Grid_t::index_t indices = {i, j, k};
        std::array<size_t, 3> nIndices = {
            {xPos.size(), yPos.size(), zPos.size()}};
        if (firstOctant) {
          // std::vectors begin with 0 and we do not want the user needing to
          // take underflow or overflow bins in account this is why we need to
          // subtract by one
          size_t m = std::abs(int(i) - (int(xPos.size())));
          size_t n = std::abs(int(j) - (int(yPos.size())));
          size_t l = std::abs(int(k) - (int(zPos.size())));

          grid.atLocalBins(indices) =
              bField.at(localToGlobalBin({{m, n, l}}, nIndices)) * BFieldUnit;

        } else {
          // std::vectors begin with 0 and we do not want the user needing to
          // take underflow or overflow bins in account this is why we need to
          // subtract by one
          grid.atLocalBins(indices) =
              bField.at(localToGlobalBin({{i - 1, j - 1, k - 1}}, nIndices)) *
              BFieldUnit;
        }
      }
    }
  }
  grid.setExteriorBins(Acts::Vector3D::Zero());

  // [3] Create the transformation for the position
  // map (x,y,z) -> (r,z)
  auto transformPos = [](const Acts::Vector3D &pos) { return pos; };

  // [4] Create the transformation for the bfield
  // map (Bx,By,Bz) -> (Bx,By,Bz)
  auto transformBField = [](const Acts::Vector3D &field,
                            const Acts::Vector3D & /*pos*/) { return field; };

  // [5] Create the mapper & BField Service
  // create field mapping
  return Acts::InterpolatedBFieldMapper<Grid_t>(
      Transform3DPos(), Transform3DBField(), std::move(grid));
}

Acts::InterpolatedBFieldMapper<
    Acts::detail::Grid<Acts::Vector2D, Acts::detail::EquidistantAxis,
                       Acts::detail::EquidistantAxis>>
Acts::solenoidFieldMapper(std::pair<ActsScalar, ActsScalar> rlim,
                          std::pair<ActsScalar, ActsScalar> zlim,
                          std::pair<size_t, size_t> nbins,
                          const SolenoidBField &field) {
  ActsScalar rMin, rMax, zMin, zMax;
  std::tie(rMin, rMax) = rlim;
  std::tie(zMin, zMax) = zlim;

  size_t nBinsR, nBinsZ;
  std::tie(nBinsR, nBinsZ) = nbins;

  ActsScalar stepZ = std::abs(zMax - zMin) / (nBinsZ - 1);
  ActsScalar stepR = std::abs(rMax - rMin) / (nBinsR - 1);

  rMax += stepR;
  zMax += stepZ;

  // Create the axis for the grid
  Acts::detail::EquidistantAxis rAxis(rMin, rMax, nBinsR);
  Acts::detail::EquidistantAxis zAxis(zMin, zMax, nBinsZ);

  // Create the grid
  using Grid_t =
      Acts::detail::Grid<Acts::Vector2D, Acts::detail::EquidistantAxis,
                         Acts::detail::EquidistantAxis>;
  Grid_t grid(std::make_tuple(std::move(rAxis), std::move(zAxis)));

  // Create the transformation for the position
  // map (x,y,z) -> (r,z)
  auto transformPos = [](const Acts::Vector3D &pos) {
    return Acts::Vector2D(perp(pos), pos.z());
  };

  // Create the transformation for the bfield
  // map (Br,Bz) -> (Bx,By,Bz)
  auto transformBField = [](const Acts::Vector2D &bfield,
                            const Acts::Vector3D &pos) {
    ActsScalar r_sin_theta_2 = pos.x() * pos.x() + pos.y() * pos.y();
    ActsScalar cos_phi, sin_phi;
    if (r_sin_theta_2 > std::numeric_limits<ActsScalar>::min()) {
      ActsScalar inv_r_sin_theta = 1. / sqrt(r_sin_theta_2);
      cos_phi = pos.x() * inv_r_sin_theta;
      sin_phi = pos.y() * inv_r_sin_theta;
    } else {
      cos_phi = 1.;
      sin_phi = 0.;
    }
    return Acts::Vector3D(bfield.x() * cos_phi, bfield.x() * sin_phi,
                          bfield.y());
  };

  // iterate over all bins, set their value to the solenoid value
  // at their lower left position
  for (size_t i = 0; i <= nBinsR + 1; i++) {
    for (size_t j = 0; j <= nBinsZ + 1; j++) {
      Grid_t::index_t index({i, j});
      if (i == 0 || j == 0 || i == nBinsR + 1 || j == nBinsZ + 1) {
        // under or overflow bin, set zero
        grid.atLocalBins(index) = Grid_t::value_type(0, 0);
      } else {
        // regular bin, get lower left boundary
        Grid_t::point_t lowerLeft = grid.lowerLeftBinEdge(index);
        // do lookup
        Vector2D B = field.getField(Vector2D(lowerLeft[0], lowerLeft[1]));
        grid.atLocalBins(index) = B;
      }
    }
  }

  // Create the mapper & BField Service
  // create field mapping
  Acts::InterpolatedBFieldMapper<Grid_t> mapper(
      Transform2DPos(), Transform2DBField(), std::move(grid));
  return mapper;
}
