#pragma once

#include "EventData/TrackParameters.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Units.hpp"

#include <fstream>
#include <iostream>
#include <random>

template <typename hits_collection_t>
void writeSimHitsObj(const hits_collection_t &simHits) {
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
void writeStatesObj(const track_state_t *states, const bool *status,
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

void writeParamsCsv(const Acts::BoundParameters *params, const bool *status,
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
