// This file is part of the Acts project.
//
// Copyright (C) 2017-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/RandomNumbers.hpp"
#include "ActsFatras/EventData/Barcode.hpp"
#include "Utilities/PdgParticle.hpp"
#include "Utilities/Units.hpp"

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
    ActsScalar phiMin = 0;
    ActsScalar phiMax = 0;
    // Low, high (inclusive) for  the longitudinal direction angle.
    //
    // This intentionally uses theta instead of eta so it can represent the
    // full direction space with finite values.
    ActsScalar thetaMin = M_PI / 2;
    ActsScalar thetaMax = M_PI / 2;
    // Low, high (exclusive) for absolute momentum.
    ActsScalar pMin = 1 * Acts::UnitConstants::GeV;
    ActsScalar pMax = 10 * Acts::UnitConstants::GeV;
    /// (Absolute) PDG particle number to identify the particle type.
    Acts::PdgParticle pdg = Acts::PdgParticle::eMuon;
    /// Randomize the charge and flip the PDG particle number sign accordingly.
    bool randomizeCharge = false;
    /// Number of particles.
    size_t numParticles = 1;
  };

  ParametricParticleGenerator(const Config &cfg);

  /// Generate a single primary vertex with the given number of particles.
  SimParticleContainer operator()(RandomEngine &rng) const;

private:
  Config m_cfg;
  // will be automatically set from PDG data tables
  ActsScalar m_charge;
  ActsScalar m_mass;
  ActsScalar m_cosThetaMin;
  ActsScalar m_cosThetaMax;
};

inline ParametricParticleGenerator::ParametricParticleGenerator(
    const Config &cfg)
    : m_cfg(cfg),
      // m_charge(ActsFatras::findCharge(m_cfg.pdg)),
      // m_mass(ActsFatras::findMass(m_cfg.pdg)),
      // since we want to draw the direction uniform on the unit sphere, we must
      // draw from cos(theta) instead of theta. see e.g.
      // https://mathworld.wolfram.com/SpherePointPicking.html
      m_cosThetaMin(std::cos(m_cfg.thetaMin)),
      // ensure upper bound is included. see e.g.
      // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
      m_cosThetaMax(std::nextafter(std::cos(m_cfg.thetaMax),
                                   std::numeric_limits<ActsScalar>::max())) {
  if (m_cfg.pdg != Acts::PdgParticle::eMuon) {
    throw std::invalid_argument("Sorry. Only eMuon is supported.");
  }
  m_charge = -1;
  m_mass = 105.6583755 * Acts::UnitConstants::MeV;
}

inline SimParticleContainer
ParametricParticleGenerator::operator()(RandomEngine &rng) const {
  using UniformIndex = std::uniform_int_distribution<unsigned int>;
  using UniformReal = std::uniform_real_distribution<ActsScalar>;

  // choose between particle/anti-particle if requested
  // the upper limit of the distribution is inclusive
  UniformIndex particleTypeChoice(0u, m_cfg.randomizeCharge ? 1u : 0u);
  // (anti-)particle choice is one random draw but defines two properties
  const Acts::PdgParticle pdgChoices[] = {
      m_cfg.pdg,
      static_cast<Acts::PdgParticle>(-m_cfg.pdg),
  };
  const ActsScalar qChoices[] = {
      m_charge,
      -m_charge,
  };
  UniformReal phiDist(m_cfg.phiMin, m_cfg.phiMax);
  UniformReal cosThetaDist(m_cosThetaMin, m_cosThetaMax);
  UniformReal pDist(m_cfg.pMin, m_cfg.pMax);

  SimParticleContainer particles;
  particles.reserve(m_cfg.numParticles);

  // counter will be reused as barcode particle number which must be non-zero.
  for (size_t ip = 1; ip <= m_cfg.numParticles; ++ip) {
    // all particles are treated as originating from the same primary vertex
    const auto pid = ActsFatras::Barcode(0u).setParticle(ip);

    // draw parameters
    const unsigned int type = particleTypeChoice(rng);
    const Acts::PdgParticle pdg = pdgChoices[type];
    const ActsScalar q = qChoices[type];
    const ActsScalar phi = phiDist(rng);
    const ActsScalar cosTheta = cosThetaDist(rng);
    const ActsScalar sinTheta = std::sqrt(1 - cosTheta * cosTheta);
    const ActsScalar p = pDist(rng);

    // we already have sin/cos theta. they can be used directly to
    Acts::Vector3D dir;
    dir[Acts::eMom0] = sinTheta * std::cos(phi);
    dir[Acts::eMom1] = sinTheta * std::sin(phi);
    dir[Acts::eMom2] = cosTheta;

    // construct the particle;
    ActsFatras::Particle particle(pid, pdg, q, m_mass);
    particle.setDirection(dir);
    particle.setAbsMomentum(p);

    // generated particle ids are already ordered and should end up at the end
    particles.emplace_back(std::move(particle));
  }

  return particles;
}

} // namespace ActsExamples
