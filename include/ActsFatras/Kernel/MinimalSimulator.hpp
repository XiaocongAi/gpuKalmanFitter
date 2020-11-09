// This file is part of the Acts project.
//
// Copyright (C) 2018-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsFatras/EventData/Hit.hpp"
#include "ActsFatras/EventData/Particle.hpp"
#include "ActsFatras/Physics/EnergyLoss/BetheBloch.hpp"
#include "ActsFatras/Physics/EnergyLoss/BetheHeitler.hpp"
#include "ActsFatras/Physics/Scattering/Highland.hpp"
#include "Surfaces/Surface.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <vector>

namespace ActsFatras {

// Measurement creator taking into account the material effects
template <typename generator_t> struct MinimalSimulator {
  // Random number generator used for the simulation.
  generator_t *generator = nullptr;
  /// Initial particle state.
  Particle particle;
  /// Highland scattering
  HighlandScattering scattering;
  /// Bethe-Bloch
  BetheBloch betheBloch;
  /// Bethe-Heitler
  BetheHeitler betheHeitler;

  struct this_result {
    /// Current/ final particle state.
    Particle particle;
    /// Material accumulated during the propagation.
    /// The initial particle can already have some accumulated material. The
    /// particle stores the full material path. This keeps track of the
    /// additional material accumulated during simulation.
    Particle::Scalar pathInX0 = 0;
    Particle::Scalar pathInL0 = 0;
    /// Whether the particle is alive or not, i.e. could be simulated further.
    bool isAlive = true;
    /// Additional particles generated by interactions.
    std::vector<Particle> generatedParticles;
    /// Hits created by the particle.
    std::vector<Hit> hits;
  };
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t &state, const stepper_t &stepper,
                  result_type &result) const {
    assert(generator and "The generator pointer must be valid");

    if (state.navigation.currentSurface == nullptr) {
      return;
    }

    const Acts::Surface &surface = *state.navigation.currentSurface;

    // avoid having a clumsy `initialized` flag by reconstructing the particle
    // state directly from the propagation state; using only the identity
    // parameters from the initial particle state.
    const auto before =
        Particle(particle)
            // include passed material from the initial particle state
            .setMaterialPassed(particle.pathInX0() + result.pathInX0,
                               particle.pathInL0() + result.pathInL0)
            .setPosition4(stepper.position(state.stepping),
                          stepper.time(state.stepping))
            .setDirection(stepper.direction(state.stepping))
            .setAbsMomentum(stepper.momentum(state.stepping));
    // we want to keep the particle state before and after the interaction.
    // since the particle is modified in-place we need a copy.
    auto after = before;

    if (surface.surfaceMaterial()) {
      // Apply global to local
      Acts::Vector2D local(0., 0.);
      surface.globalToLocal(state.options.geoContext,
                            stepper.position(state.stepping),
                            stepper.direction(state.stepping), local);

      Acts::MaterialSlab slab = surface.surfaceMaterial().materialSlab(local);

      if (slab) {
        // adapt material for non-zero incidence
        auto normal = surface.normal(state.geoContext, local);
        // dot-product(unit normal, direction) = cos(incidence angle)
        // particle direction is normalized, not sure about surface normal
        auto cosIncidenceInv =
            normal.norm() / normal.dot(before.unitDirection());
        slab.scaleThickness(cosIncidenceInv);

        // The place where the material has effects on the position/direction of
        // the 'after'
        // @note No children generated for the moment
        scattering(*generator, slab, after);
        betheBloch(*generator, slab, after);
        betheHeitler(*generator, slab, after);

        // add the accumulated material; assumes the full material was passsed
        // event if the particle was killed.
        result.pathInX0 += slab.thicknessInX0();
        result.pathInL0 += slab.thicknessInL0();
        // WARNING this overwrites changes that the physics interactions
        //         might have performed with regard to the passed material.
        //         ensures consistent material counting by making the one
        //         component that by construction will see all material
        //         contributions (this Interactor) responsible.
        // TODO review this for supporting multiple interactions within the
        // same material slab
        after.setMaterialPassed(before.pathInX0() + slab.thicknessInX0(),
                                before.pathInL0() + slab.thicknessInL0());
      }
    }

    // store results of this interaction step, including potential hits
    result.particle = after;
    result.hits.emplace_back(
        surface.geoID(), before.particleId(),
        // the interaction could potentially modify the particle position
        Hit::Scalar(0.5) * (before.position4() + after.position4()),
        before.momentum4(), after.momentum4(), result.hits.size());

    // continue the propagation with the modified parameters
    stepper.update(state.stepping, after.position(), after.unitDirection(),
                   after.absMomentum(), after.time());
  }
};

} // namespace ActsFatras
