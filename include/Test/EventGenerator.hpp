#include "ActsExamples/Generator.hpp"
#include "ActsFatras/Particle.hpp"
#include "ActsFatras/Barcode.hpp"
#include "EventData/TrackParameters.hpp"
#include "Utilities/ParameterDefinitions.hpp"

#include <iostream>
#include <string>
#include <vector>


using SimParticleContainer = std::vector<ActsFatras::Particle>;

template <typename random_engine_t>
void runGeneration(const random_engine_t& rng, const ActsExamples::Generator& generator, SimParticleContainer& particles){
   size_t nPrimaryVertices = 0;  
  // generate the primary vertices from this generator
    for (size_t n = generate.multiplicity(rng); 0 < n; --n) {
      nPrimaryVertices += 1;

      // generate primary vertex position
      auto vertexPosition = generate.vertex(rng);
      // generate particles associated to this vertex
      auto vertexParticles = generate.particles(rng);
      
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

template < typename propagator_t, typename propagator_options_t, typename sim_hits_t>
 void runSimulation(const propagator_t& propagator, const propagator_options_t& propOptions, const SimParticleContainer& particles, sim_hits_t& simHits){

  //std::cout << "Start to run propagation to create measurements" << std::endl;
  //auto start_propagate = std::chrono::high_resolution_clock::now();

// Run propagation to create the measurements
// @todo The material effects have to be considered during the simulation
  for (const auto& particle: particles) {
    
    // use AnyCharge to be able to handle neutral and charged parameters
    Acts::CurvilinearParameters start(Acts::BoundSymMatrix::Zero(), particle.position(), particle.unitDirection()*particle.absMomentum(),  particle.charge(), particle.time());

    propagator.propagate(start, propOptions, simHits[ip]);
 }

  //auto end_propagate = std::chrono::high_resolution_clock::now();
  //std::chrono::duration<double> elapsed_seconds =
  //    end_propagate - start_propagate;
  //std::cout << "Time (ms) to run propagation tests: "
  //          << elapsed_seconds.count() * 1000 << std::endl;
}

