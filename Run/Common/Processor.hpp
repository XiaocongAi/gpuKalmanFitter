#pragma once

#include "FitData.hpp"
#include "Utilities/detail/periodic.hpp"

#include "Test/Helper.hpp"

#include <array>
#include <iostream>
#include <random>
#include <vector>

using SimParticleContainer = std::vector<ActsFatras::Particle>;
using SimResultContainer = std::vector<Simulator::result_type>;
using ParametersContainer =
    std::vector<Acts::BoundParameters<Acts::LineSurface>>;
using TargetSurfaceContainer = std::vector<Acts::LineSurface>;

struct ParticleSmearingParameters {
  /// Constant term of the d0 resolution.
  float sigmaD0 = 30 * Acts::units::_um;
  /// Pt-dependent d0 resolution of the form sigma_d0 = A*exp(-1.*abs(B)*pt).
  float sigmaD0PtA = 0 * Acts::units::_um;
  float sigmaD0PtB = 1 / Acts::units::_GeV;
  /// Constant term of the z0 resolution.
  float sigmaZ0 = 30 * Acts::units::_um;
  /// Pt-dependent z0 resolution of the form sigma_z0 = A*exp(-1.*abs(B)*pt).
  float sigmaZ0PtA = 0 * Acts::units::_um;
  float sigmaZ0PtB = 1 / Acts::units::_GeV;
  /// Time resolution.
  float sigmaT0 = 5 * Acts::units::_ns;
  /// Phi angular resolution.
  float sigmaPhi = 1 * Acts::UnitConstants::degree;
  /// Theta angular resolution.
  float sigmaTheta = 1 * Acts::UnitConstants::degree;
  /// Relative momentum resolution.
  float sigmaPRel = 0.001;
};

template <typename random_engine_t>
void runParticleGeneration(random_engine_t &rng,
                           const ActsExamples::Generator &generator,
                           SimParticleContainer &particles) {
  size_t nPrimaryVertices = 0;
  // generate the primary vertices from this generator
  for (size_t n = generator.multiplicity(rng); 0 < n; --n) {
    nPrimaryVertices += 1;

    // generate primary vertex position
    auto vertexPosition = generator.vertex(rng);
    // generate particles associated to this vertex
    auto vertexParticles = generator.particles(rng);

    auto updateParticleInPlace = [&](ActsFatras::Particle &particle) {
      // only set the primary vertex, leave everything else as-is
      // using the number of primary vertices as the index ensures
      // that barcode=0 is not used, since it is used elsewhere
      // to signify elements w/o an associated particle.
      const auto pid = ActsFatras::Barcode(particle.particleId())
                           .setVertexPrimary(nPrimaryVertices);
      // move particle to the vertex
      const auto pos4 = (vertexPosition + particle.position4()).eval();
      // `withParticleId` returns a copy because it changes the identity
      particle = particle.withParticleId(pid).setPosition4(pos4);
    };
    for (auto &vertexParticle : vertexParticles) {
      updateParticleInPlace(vertexParticle);
    }
    // copy to particles collection
    std::copy(vertexParticles.begin(), vertexParticles.end(),
              std::back_inserter(particles));
  }
}

template <typename random_engine_t, typename propagator_t>
void runSimulation(const Acts::GeometryContext &gctx,
                   const Acts::MagneticFieldContext &mctx, random_engine_t &rng,
                   const propagator_t &propagator,
                   const SimParticleContainer &generatedParticles,
                   SimParticleContainer &validParticles,
                   SimResultContainer &simResults,
                   const Acts::Surface *surfaces, size_t nSurfaces) {
  size_t ip = 0;

  for (const auto &particle : generatedParticles) {
    if (ip < validParticles.size()) {
      // Construct a propagator options for each propagate
      PropOptionsType propOptions(gctx, mctx);
      propOptions.initializer.surfaceSequence = surfaces;
      propOptions.initializer.surfaceSequenceSize = nSurfaces;
      propOptions.absPdgCode = particle.pdg();
      propOptions.mass = particle.mass();
      propOptions.action.generator = &rng;
      propOptions.action.particle = particle;
      Acts::CurvilinearParameters start(
          Acts::BoundSymMatrix::Zero(), particle.position(),
          particle.unitDirection() * particle.absMomentum(), particle.charge(),
          particle.time());
      Simulator::result_type simResult;
      propagator.propagate(start, propOptions, simResult);
      // The particles must have nSurfaces sim hits. Otherwise, skip this
      // simulation result
      if (simResult.hits.size() != nSurfaces) {
        std::cout << "Warning! Generated particle rejected!" << std::endl;
        continue;
      }
      // store the sim particles and hits
      validParticles[ip] = particle;
      simResults[ip] = simResult;
      ip++;
    }
  }
  // In case we are not able to get the same size of simulated particles as
  // requested
  if (ip < validParticles.size()) {
    throw std::runtime_error(
        "Too many generated particles rejected! Simulation failed!\n");
  }
}

void buildTargetSurfaces(const SimParticleContainer &validParticles,
                         Acts::LineSurface *targetSurfaces) {
  // Write directly into the container
  unsigned int ip = 0;
  for (const auto &particle : validParticles) {
    targetSurfaces[ip] = Acts::LineSurface(particle.position());
    ip++;
  }
}

// @note using concreate surface type to avoid trivial advance of the
// Acts::Surface* to the PlaneSurfaceType* as in the DirectNavigator
template <typename random_engine_t>
void runHitSmearing(const Acts::GeometryContext &gctx, random_engine_t &rng,
                    const SimResultContainer &simResults,
                    const std::array<float, 2> &resolution,
                    Acts::PixelSourceLink *sourcelinks,
                    const PlaneSurfaceType *surfaces, size_t nSurfaces) {
  // The normal dist
  std::normal_distribution<float> stdNormal(0.0, 1.0);
  // Perform smearing to the simulated hits
  for (int ip = 0; ip < simResults.size(); ip++) {
    auto hits = simResults[ip].hits;
    auto nHits = hits.size();
    if (nHits != nSurfaces) {
      throw std::invalid_argument("Sim hits size should be exactly" +
                                  nSurfaces);
    }
    for (unsigned int ih = 0; ih < nHits; ih++) {
      // Apply global to local
      Acts::Vector2D lPos;
      // find the surface for this hit
      // @note Using operator[] to get the object might be dangerous if there is
      // implicit type conversion of the pointer
      surfaces[ih].globalToLocal(gctx, hits[ih].position(),
                                 hits[ih].unitDirection(), lPos);
      // Perform the smearing to truth
      float dx = resolution[0] * stdNormal(rng);
      float dy = resolution[1] * stdNormal(rng);

      // The measurement values
      Acts::Vector2D values;
      values << lPos[0] + dx, lPos[1] + dy;

      // The measurement covariance
      Acts::SymMatrix2D cov;
      cov << resolution[0] * resolution[0], 0., 0.,
          resolution[1] * resolution[1];

      // Push back to the container
      sourcelinks[ip * nSurfaces + ih] =
          Acts::PixelSourceLink(values, cov, &surfaces[ih]);
    }
  }
}

template <typename random_engine_t>
ParametersContainer
runParticleSmearing(random_engine_t &rng, const Acts::GeometryContext &gctx,
                    const SimParticleContainer &validParticles,
                    const ParticleSmearingParameters &resolution,
                    const Acts::LineSurface *targetSurfaces, size_t nTracks) {
  if (validParticles.size() != nTracks) {
    std::runtime_error("validParticles size not equal to number of tracks!");
  }

  // The normal dist
  std::normal_distribution<float> stdNormal(0.0, 1.0);

  // Reserve the container
  ParametersContainer parameters;
  parameters.reserve(validParticles.size());

  // Perform smearing to the sim particles
  unsigned int ip = 0;
  for (const auto particle : validParticles) {
    const auto time = particle.time();
    const auto phi = Acts::VectorHelpers::phi(particle.unitDirection());
    const auto theta = Acts::VectorHelpers::theta(particle.unitDirection());
    const auto pt = particle.transverseMomentum();
    const auto p = particle.absMomentum();
    const auto q = particle.charge();

    // compute momentum-dependent resolutions
    const float sigmaD0 =
        resolution.sigmaD0 +
        resolution.sigmaD0PtA *
            std::exp(-1.0 * std::abs(resolution.sigmaD0PtB) * pt);
    const float sigmaZ0 =
        resolution.sigmaZ0 +
        resolution.sigmaZ0PtA *
            std::exp(-1.0 * std::abs(resolution.sigmaZ0PtB) * pt);
    const float sigmaP = resolution.sigmaPRel * p;
    // var(q/p) = (d(1/p)/dp)² * var(p) = (-1/p²)² * var(p)
    const float sigmaQOverP = sigmaP / (p * p);
    // shortcuts for other resolutions
    const float sigmaT0 = resolution.sigmaT0;
    const float sigmaPhi = resolution.sigmaPhi;
    const float sigmaTheta = resolution.sigmaTheta;

    Acts::BoundVector params = Acts::BoundVector::Zero();
    // smear the position/time
    params[Acts::eBoundLoc0] = sigmaD0 * stdNormal(rng);
    params[Acts::eBoundLoc1] = sigmaZ0 * stdNormal(rng);
    params[Acts::eBoundTime] = time + sigmaT0 * stdNormal(rng);
    // smear direction angles phi,theta ensuring correct bounds
    const auto phiTheta = Acts::detail::normalizePhiTheta(
        phi + sigmaPhi * stdNormal(rng), theta + sigmaTheta * stdNormal(rng));
    const float newPhi = phiTheta.first;
    const float newTheta = phiTheta.second;
    params[Acts::eBoundPhi] = newPhi;
    params[Acts::eBoundTheta] = newTheta;
    // compute smeared absolute momentum vector
    const float newP = std::max(0.0, p + sigmaP * stdNormal(rng));
    params[Acts::eBoundQOverP] = (q != 0) ? (q / newP) : (1 / newP);

    // build the track covariance matrix using the smearing sigmas
    Acts::BoundSymMatrix cov = Acts::BoundSymMatrix::Zero();
    cov(Acts::eBoundLoc0, Acts::eBoundLoc0) = sigmaD0 * sigmaD0;
    cov(Acts::eBoundLoc1, Acts::eBoundLoc1) = sigmaZ0 * sigmaZ0;
    cov(Acts::eBoundTime, Acts::eBoundTime) = sigmaT0 * sigmaT0;
    cov(Acts::eBoundPhi, Acts::eBoundPhi) = sigmaPhi * sigmaPhi;
    cov(Acts::eBoundTheta, Acts::eBoundTheta) = sigmaTheta * sigmaTheta;
    cov(Acts::eBoundQOverP, Acts::eBoundQOverP) = sigmaQOverP * sigmaQOverP;

    // Construct a bound parameters with a perigee surface as the reference
    // surface
    parameters.emplace_back(gctx, cov, params, &targetSurfaces[ip]);
    ip++;
  }
  return parameters;
}
