// This file is part of the Acts project.
//
// Copyright (C) 2017-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Utilities/PdgParticle.hpp"
#include "Utilities/Units.hpp"
#include "ActsExamples/RandomNumbers.hpp"

#include <array>
#include <cmath>

namespace ActsExamples {
/// The sim particle container
using SimParticleContainer = std::vector<ActsFatras::Particle>;

/// Generate particles from uniform parameter distributions.
///
/// Generates a single vertex with the given number of tracks. The track
/// direction is drawn from a uniform distribution on the unit sphere (within
/// the given limits). Its absolute momentum is drawn from a uniform
/// distribution. Position and time are always set to zero.
class ParametricParticleGenerator {
 public:
  struct Config {
    // Low, high (exclusive) for the transverse direction angle.
    double phiMin = -M_PI;
    double phiMax = M_PI;
    // Low, high (inclusive) for  the longitudinal direction angle.
    //
    // This intentionally uses theta instead of eta so it can represent the
    // full direction space with finite values.
    double thetaMin = 0.0;
    double thetaMax = M_PI;
    // Low, high (exclusive) for absolute momentum.
    double pMin = 1 * Acts::UnitConstants::GeV;
    double pMax = 10 * Acts::UnitConstants::GeV;
    /// (Absolute) PDG particle number to identify the particle type.
    Acts::PdgParticle pdg = Acts::PdgParticle::eMuon;
    /// Randomize the charge and flip the PDG particle number sign accordingly.
    bool randomizeCharge = false;
    /// Number of particles.
    size_t numParticles = 1;
  };

  ParametricParticleGenerator(const Config& cfg);

  /// Generate a single primary vertex with the given number of particles.
  SimParticleContainer operator()(RandomEngine& rng) const;

 private:
  Config m_cfg;
  // will be automatically set from PDG data tables
  double m_charge;
  double m_mass;
  double m_cosThetaMin;
  double m_cosThetaMax;
};

}  // namespace ActsExamples
