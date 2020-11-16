#pragma once

#include "EventData/TrackParameters.hpp"
#include "EventData/detail/coordinate_transformations.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Helpers.hpp"
#include "Utilities/Units.hpp"

#include <fstream>
#include <iostream>
#include <random>

#include <TFile.h>
#include <TTree.h>

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

template <typename parameters_t>
void writeParamsCsv(const parameters_t *params, const bool *status,
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

// Write the fitted parameters and truth particle info (one entry for one track)
template <typename parameters_t>
void writeParamsRoot(const Acts::GeometryContext &gctx,
                     const parameters_t *fittedParams, const bool *status,
                     const SimParticleContainer &simParticles,
                     unsigned int nTracks, std::string fileName,
                     std::string treeName) {
  // Define the variables to write out
  int t_charge{0};
  float t_time{0};
  float t_vx{-99.};
  float t_vy{-99.};
  float t_vz{-99.};
  float t_px{-99.};
  float t_py{-99.};
  float t_pz{-99.};
  float t_theta{-99.};
  float t_phi{-99.};
  float t_pT{-99.};
  float t_eta{-99.};
  std::array<float, 6> params_fit = {-99, -99, -99, -99, -99, -99};
  std::array<float, 6> err_params_fit = {-99, -99, -99, -99, -99, -99};
  std::array<float, 6> res_params = {-99, -99, -99, -99, -99, -99};
  std::array<float, 6> pull_params = {-99, -99, -99, -99, -99, -99};

  // Create the root file
  TFile *file = TFile::Open(fileName.c_str(), "RECREATE");
  if (file == nullptr) {
    throw std::ios_base::failure("Could not open " + fileName);
  }
  file->cd();
  // Create the root tree
  TTree *tree = new TTree(treeName.c_str(), treeName.c_str());
  if (tree == nullptr)
    throw std::bad_alloc();
  else {
    tree->Branch("t_charge", &t_charge);
    tree->Branch("t_time", &t_time);
    tree->Branch("t_vx", &t_vx);
    tree->Branch("t_vy", &t_vy);
    tree->Branch("t_vz", &t_vz);
    tree->Branch("t_px", &t_px);
    tree->Branch("t_py", &t_py);
    tree->Branch("t_pz", &t_pz);
    tree->Branch("t_theta", &t_theta);
    tree->Branch("t_phi", &t_phi);
    tree->Branch("t_eta", &t_eta);
    tree->Branch("t_pT", &t_pT);

    tree->Branch("eLOC0_fit", &params_fit[0]);
    tree->Branch("eLOC1_fit", &params_fit[1]);
    tree->Branch("ePHI_fit", &params_fit[2]);
    tree->Branch("eTHETA_fit", &params_fit[3]);
    tree->Branch("eQOP_fit", &params_fit[4]);
    tree->Branch("eT_fit", &params_fit[5]);

    tree->Branch("err_eLOC0_fit", &err_params_fit[0]);
    tree->Branch("err_eLOC1_fit", &err_params_fit[1]);
    tree->Branch("err_ePHI_fit", &err_params_fit[2]);
    tree->Branch("err_eTHETA_fit", &err_params_fit[3]);
    tree->Branch("err_eQOP_fit", &err_params_fit[4]);
    tree->Branch("err_eT_fit", &err_params_fit[5]);

    tree->Branch("res_eLOC0", &res_params[0]);
    tree->Branch("res_eLOC1", &res_params[1]);
    tree->Branch("res_ePHI", &res_params[2]);
    tree->Branch("res_eTHETA", &res_params[3]);
    tree->Branch("res_eQOP", &res_params[4]);
    tree->Branch("res_eT", &res_params[5]);

    tree->Branch("pull_eLOC0", &pull_params[0]);
    tree->Branch("pull_eLOC1", &pull_params[1]);
    tree->Branch("pull_ePHI", &pull_params[2]);
    tree->Branch("pull_eTHETA", &pull_params[3]);
    tree->Branch("pull_eQOP", &pull_params[4]);
    tree->Branch("pull_eT", &pull_params[5]);
  }

  // Fill the tree
  for (unsigned int it = 0; it < nTracks; it++) {
    // we skip the unsuccessful tracks
    if (not status[it]) {
      continue;
    }
    const auto &fitBoundParams = fittedParams[it];
    const auto &parameters = fitBoundParams.parameters();
    const auto &covariance = *fitBoundParams.covariance();

    // The truth particle info
    const auto &particle = simParticles[it];
    const auto p = particle.absMomentum();
    t_charge = particle.charge();
    t_time = particle.time();
    t_vx = particle.position().x();
    t_vy = particle.position().y();
    t_vz = particle.position().z();
    t_px = p * particle.unitDirection().x();
    t_py = p * particle.unitDirection().y();
    t_pz = p * particle.unitDirection().z();
    t_theta = Acts::VectorHelpers::theta(particle.unitDirection());
    t_phi = Acts::VectorHelpers::phi(particle.unitDirection());
    t_eta = Acts::VectorHelpers::eta(particle.unitDirection());
    t_pT = p * Acts::VectorHelpers::perp(particle.unitDirection());

    Acts::BoundVector truthParameters;
    truthParameters << 0, 0, t_phi, t_theta, t_charge / p, t_time;
    for (unsigned int ip = 0; ip < Acts::eBoundParametersSize; ip++) {
      params_fit[ip] = parameters[ip];
      err_params_fit[ip] = sqrt(covariance(ip, ip));
      res_params[ip] = parameters[ip] - truthParameters[ip];
      pull_params[ip] = res_params[ip] / err_params_fit[ip];
    }
    tree->Fill();
  }

  file->cd();
  tree->Write();
  file->Close();
}
