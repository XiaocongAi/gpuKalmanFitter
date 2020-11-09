#pragma once

#include "ActsExamples/RandomNumbers.hpp"
#include "ActsExamples/Generator.hpp"

#include "ActsFatras/EventData/Particle.hpp"
#include "ActsFatras/EventData/Barcode.hpp"
#include "ActsFatras/Kernel/MinimalSimulator.hpp"

#include "EventData/TrackParameters.hpp"
#include "Geometry/GeometryContext.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/detail/periodic.hpp"
#include "Utilities/Units.hpp"
#include "Utilities/Definitions.hpp"

#include <vector>
#include <array>
#include <random>

using Simulator = ActsFatras::MinimalSimulator<ActsExamples::RandomEngine>;
using SimParticleContainer = std::vector<ActsFatras::Particle>;
using SimResultContainer = std::vector<Simulator::result_type>;
using ParametersContainer = std::vector<Acts::CurvilinearParameters>;

struct ParticleSmearingParameters{
    /// Constant term of the d0 resolution.
    double sigmaD0 = 30 * Acts::units::_um;
    /// Pt-dependent d0 resolution of the form sigma_d0 = A*exp(-1.*abs(B)*pt).
    double sigmaD0PtA = 0 * Acts::units::_um;
    double sigmaD0PtB = 1 / Acts::units::_GeV;
    /// Constant term of the z0 resolution.
    double sigmaZ0 = 30 * Acts::units::_um;
    /// Pt-dependent z0 resolution of the form sigma_z0 = A*exp(-1.*abs(B)*pt).
    double sigmaZ0PtA = 0 * Acts::units::_um;
    double sigmaZ0PtB = 1 / Acts::units::_GeV;
    /// Time resolution.  
    double sigmaT0 = 5 * Acts::units::_ns;
    /// Phi angular resolution.
    double sigmaPhi = 1 * Acts::UnitConstants::degree;
    /// Theta angular resolution.
    double sigmaTheta = 1 * Acts::UnitConstants::degree;
    /// Relative momentum resolution.
    double sigmaPRel = 0.001;
    
};

template <typename random_engine_t>
void runGeneration(random_engine_t& rng, const ActsExamples::Generator& generator, SimParticleContainer& particles){
   size_t nPrimaryVertices = 0;  
  // generate the primary vertices from this generator
    for (size_t n = generator.multiplicity(rng); 0 < n; --n) {
      nPrimaryVertices += 1;

      // generate primary vertex position
      auto vertexPosition = generator.vertex(rng);
      // generate particles associated to this vertex
      auto vertexParticles = generator.particles(rng);
      
       auto updateParticleInPlace = [&](ActsFatras::Particle& particle) {
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
      for (auto& vertexParticle : vertexParticles) {
        updateParticleInPlace(vertexParticle);
      }
      // copy to particles collection
       std::copy(vertexParticles.begin(), vertexParticles.end(),
              std::back_inserter(particles));  
   }  
} 

template < typename propagator_t, typename propagator_options_t>
 void runSimulation(const propagator_t& propagator, const propagator_options_t& propOptions, const SimParticleContainer& particles, SimResultContainer& simResults){
  size_t ip=0; 
  for (const auto& particle: particles) {
    Acts::CurvilinearParameters start(Acts::BoundSymMatrix::Zero(), particle.position(), particle.unitDirection()*particle.absMomentum(),  particle.charge(), particle.time());
    propagator.propagate(start, propOptions, simResults[ip]);
    ip++; 
 }
}

template <typename random_engine_t>
void runHitSmearing(random_engine_t& rng, const Acts::GeometryContext& gctx, const SimResultContainer& simResults, const std::array<double, 2>& resolution, Acts::PixelSourceLink* sourcelinks, const Acts::Surface* surfaces, size_t nSurfaces){
// The normal dist 
std::normal_distribution<double> stdNormal(0.0, 1.0); 
// Perform smearing to the simulated hits
 for (int ip = 0; ip < simResults.size(); ip++) {
   auto hits = simResults[ip].hits;
   auto nHits = hits.size();
   if(nHits!=nSurfaces){ 
     throw std::runtime_error("Sim hits size should be exactly" + nSurfaces); 
   } 
   for (unsigned int ih = 0; ih < nHits; ih++) {
     // Apply global to local
     Acts::Vector2D lPos;
     // find the surface for this hit
     surfaces[ih].globalToLocal(gctx, hits[ih].position(),
                                hits[ih].unitDirection(), lPos);
     // Perform the smearing to truth
     double dx = resolution[0]*stdNormal(rng);
     double dy = resolution[1]*stdNormal(rng);

     // The measurement values
     Acts::Vector2D values;
     values << lPos[0] + dx, lPos[1] + dy;

     // The measurement covariance
     Acts::SymMatrix2D cov;
     cov << resolution[0]*resolution[0], 0., 0., resolution[1]*resolution[1];

     // Push back to the container
     sourcelinks[ip * nSurfaces + ih] =
         Acts::PixelSourceLink(values, cov, &surfaces[ih]);
   }
 }
}

template <typename random_engine_t>
ParametersContainer runParticleSmearing(random_engine_t& rng, const Acts::GeometryContext& gctx, const SimParticleContainer& particles, const ParticleSmearingParameters& resolution, size_t nParticles){
// The normal dist 
std::normal_distribution<double> stdNormal(0.0, 1.0); 

// Reserve the container
ParametersContainer parameters;
parameters.reserve(nParticles);

 // Perform smearing to the sim particles
 for (const auto particle: particles) {
      const auto time = particle.time();
      const auto phi = Acts::VectorHelpers::phi(particle.unitDirection());
      const auto theta = Acts::VectorHelpers::theta(particle.unitDirection());
      const auto pt = particle.transverseMomentum();
      const auto p = particle.absMomentum();
      const auto q = particle.charge();
      // compute momentum-dependent resolutions
      const double sigmaD0 =
          resolution.sigmaD0 +
          resolution.sigmaD0PtA * std::exp(-1.0 * std::abs(resolution.sigmaD0PtB) * pt);
      const double sigmaZ0 =
          resolution.sigmaZ0 +
          resolution.sigmaZ0PtA * std::exp(-1.0 * std::abs(resolution.sigmaZ0PtB) * pt);
      const double sigmaP = resolution.sigmaPRel * p;
      // var(q/p) = (d(1/p)/dp)² * var(p) = (-1/p²)² * var(p)
      const double sigmaQOverP = sigmaP / (p * p);
      // shortcuts for other resolutions
      const double sigmaT0 = resolution.sigmaT0;
      const double sigmaPhi = resolution.sigmaPhi;
      const double sigmaTheta = resolution.sigmaTheta;

      Acts::BoundVector params = Acts::BoundVector::Zero();
      // smear the position/time
      params[Acts::eBoundLoc0] = sigmaD0 * stdNormal(rng);
      params[Acts::eBoundLoc1] = sigmaZ0 * stdNormal(rng);
      params[Acts::eBoundTime] = time + sigmaT0 * stdNormal(rng);
      // smear direction angles phi,theta ensuring correct bounds
      const auto phiTheta = Acts::detail::normalizePhiTheta(
          phi + sigmaPhi * stdNormal(rng), theta + sigmaTheta * stdNormal(rng));
      const double newPhi = phiTheta.first;
      const double newTheta = phiTheta.second;
      params[Acts::eBoundPhi] = newPhi;
      params[Acts::eBoundTheta] = newTheta;
      // compute smeared absolute momentum vector
      const double newP = std::max(0.0, p + sigmaP * stdNormal(rng));
      params[Acts::eBoundQOverP] = (q != 0) ? (q / newP) : (1 / newP);

      // build the track covariance matrix using the smearing sigmas
      Acts::BoundSymMatrix cov = Acts::BoundSymMatrix::Zero();
      cov(Acts::eBoundLoc0, Acts::eBoundLoc0) = sigmaD0 * sigmaD0;
      cov(Acts::eBoundLoc1, Acts::eBoundLoc1) = sigmaZ0 * sigmaZ0;
      cov(Acts::eBoundTime, Acts::eBoundTime) = sigmaT0 * sigmaT0;
      cov(Acts::eBoundPhi, Acts::eBoundPhi) = sigmaPhi * sigmaPhi;
      cov(Acts::eBoundTheta, Acts::eBoundTheta) = sigmaTheta * sigmaTheta;
      cov(Acts::eBoundQOverP, Acts::eBoundQOverP) = sigmaQOverP * sigmaQOverP;

      Acts::Vector3D newDir(sin(newTheta)*cos(newPhi), sin(newTheta)*sin(newPhi), cos(newTheta));
      Acts::Vector3D newMom = newDir*newP;
      Acts::Vector3D measY(0.,0.,1.);
      Acts::Vector3D measX = measY.cross(newDir);
      Acts::Vector3D shift = params[Acts::eBoundLoc0]*measX + params[Acts::eBoundLoc1]*measY;
      Acts::Vector3D newPos = particle.position() + shift;
     parameters.emplace_back(cov, newPos, newMom, q, params[Acts::eBoundTime]); 
 }
 return parameters;
}
