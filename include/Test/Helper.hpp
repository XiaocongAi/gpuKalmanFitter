#pragma once

#include "EventData/PixelSourceLink.hpp"
#include "EventData/TrackParameters.hpp"
#include "Material/Material.hpp"
#include "Material/MaterialSlab.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Units.hpp"

#include <fstream>
#include <iostream>
#include <random>

namespace Test {

// Struct for B field
struct ConstantBField {
  ACTS_DEVICE_FUNC static Acts::Vector3D
  getField(const Acts::Vector3D & /*field*/) {
    return Acts::Vector3D(0., 0., 2. * Acts::units::_T);
  }
};

// Silicon material
Acts::Material makeSilicon() {
  return Acts::Material::fromMolarDensity(
      9.370 * Acts::units::_cm, 46.52 * Acts::units::_cm, 28.0855, 14,
      (2.329 / 28.0855) * Acts::UnitConstants::mol / Acts::UnitConstants::cm3);
}

// Measurement creator
template <typename generator_t> struct MeasurementCreator {
  /// Random number generator used for the simulation.
  generator_t *generator = nullptr;

  // The smearing resolution
  double resX = 30 * Acts::units::_um;
  double resY = 30 * Acts::units::_um;

  struct this_result {
    std::vector<Acts::PixelSourceLink> sourcelinks;
  };
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t &state, const stepper_t &stepper,
                  result_type &result) const {
    assert(generator and "The generator pointer must be valid");

    if (state.navigation.currentSurface != nullptr) {
      // Apply global to local
      Acts::Vector2D lPos;
      state.navigation.currentSurface->globalToLocal(
          state.options.geoContext, stepper.position(state.stepping),
          stepper.direction(state.stepping), lPos);
      // Perform the smearing to truth
      double dx = std::normal_distribution<double>(0., resX)(*generator);
      double dy = std::normal_distribution<double>(0., resY)(*generator);

      // The measurement values
      Acts::Vector2D values;
      values << lPos[0] + dx, lPos[1] + dy;

      // The measurement covariance
      Acts::SymMatrix2D cov;
      cov << resX * resX, 0., 0., resY * resY;

      // Push back to the container
      result.sourcelinks.emplace_back(values, cov,
                                      state.navigation.currentSurface);
    }
    return;
  }
};

// Test actor
struct VoidActor {
  struct this_result {
    bool status = false;
  };
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  ACTS_DEVICE_FUNC void operator()(propagator_state_t &state,
                                   const stepper_t &stepper,
                                   result_type &result) const {
    return;
  }
};

// Test aborter
struct VoidAborter {
  template <typename propagator_state_t, typename stepper_t, typename result_t>
  ACTS_DEVICE_FUNC bool operator()(propagator_state_t &state,
                                   const stepper_t &stepper,
                                   result_t &result) const {
    return false;
  }
};

template <typename hits_collection_t>
void writeSimHits(const hits_collection_t &simHits) {
  // Write all of the created tracks to one obj file
  std::ofstream obj_hits;
  std::string fileName = "sim-hits.obj";
  obj_hits.open(fileName.c_str());

  // Initialize the vertex counter
  unsigned int vCounter = 0;
  for (unsigned int ih = 0; ih < simHits.size(); ih++) {
    auto hits = simHits[ih].hits;
    ++vCounter;
    for (const auto &sl : hits) {
      const auto &pos = sl.position();
      obj_hits << "v " << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
    }
    // Write out the line - only if we have at least two points created
    size_t vBreak = vCounter + hits.size() - 1;
    for (; vCounter < vBreak; ++vCounter)
      obj_hits << "l " << vCounter << " " << vCounter + 1 << '\n';
  }
  obj_hits.close();
}

template <typename track_state_t>
void writeStates(const track_state_t *states, const bool *status,
                 unsigned int nTracks, unsigned int nSurfaces,
                 std::string fileName, std::string parameters = "smoothed") {
  // Write all of the created tracks to one obj file
  std::ofstream obj_tracks;
  if (fileName.empty()) {
    fileName = "tracks-fitted.obj";
  }
  obj_tracks.open(fileName.c_str());

  // Initialize the vertex counter
  unsigned int vCounter = 0;
  for (unsigned int it = 0; it < nTracks; it++) {
    // we skip the unsuccessful tracks
    if (not status[it]) {
      continue;
    }
    ++vCounter;
    for (int is = 0; is < nSurfaces; is++) {
      Acts::Vector3D pos;
      if (parameters == "predicted") {
        pos = states[it * nSurfaces + is].parameter.predicted.position();
      } else if (parameters == "filtered") {
        pos = states[it * nSurfaces + is].parameter.filtered.position();
      } else {
        pos = states[it * nSurfaces + is].parameter.smoothed.position();
      }
      obj_tracks << "v " << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
    }
    // Write out the line - only if we have at least two points created
    size_t vBreak = vCounter + nSurfaces - 1;
    for (; vCounter < vBreak; ++vCounter)
      obj_tracks << "l " << vCounter << " " << vCounter + 1 << '\n';
  }
  obj_tracks.close();
}

void writeParams(const Acts::BoundParameters *params, const bool *status,
                 unsigned int nTracks, std::string fileName) {
  // Write all of the created tracks to one obj file
  std::ofstream csv_params;
  if (fileName.empty()) {
    fileName = "params-fitted.obj";
  }
  csv_params.open(fileName.c_str());

  // write the csv header
  csv_params << "trackIdx, fit_BoundLoc0, fit_BoundLoc1, fit_BoundPhi, "
                "fit_BoundTheta, fit_BoundQOverP, fit_BoundTime"
             << '\n';
  // Initialize the t counter
  for (unsigned int it = 0; it < nTracks; it++) {
    // we skip the unsuccessful tracks
    if (not status[it]) {
      continue;
    }
    const auto parameters = params[it].parameters();
    csv_params << it << "," << parameters[Acts::eBoundLoc0] << ","
               << parameters[Acts::eBoundLoc1] << ","
               << parameters[Acts::eBoundPhi] << ","
               << parameters[Acts::eBoundTheta] << ","
               << parameters[Acts::eBoundQOverP] << ","
               << parameters[Acts::eBoundTime] << '\n';
  }
  csv_params.close();
}

} // namespace Test
