// This file is part of the Acts project.
//
// Copyright (C) 2019-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/// @file
/// @date 2018-03-13
/// @author Moritz Kiehn <msmk@cern.ch>

#pragma once

#include "ActsExamples/RandomNumbers.hpp"

namespace ActsExamples {
/// The sim particle container
using SimParticleContainer = std::vector<ActsFatras::Particle>;

 /// Combined set of generator functions.
  ///
  /// Each generator creates a number of primary vertices (multiplicity),
  /// each with an separate vertex position and time (vertex), and a set of
  /// associated particles grouped into secondary vertices (process) anchored
  /// at the primary vertex position. The first group of particles generated
  /// by the process are the particles associated directly to the primary
  /// vertex.
  ///
  /// The process generator is responsible for defining all components of the
  /// particle barcode except the primary vertex. The primary vertex will be
  /// set/overwritten by the event generator.
  using MultiplicityGenerator = std::function<size_t(RandomEngine&)>;
  using VertexGenerator = std::function<Acts::Vector4D(RandomEngine&)>;
  using ParticlesGenerator = std::function<SimParticleContainer(RandomEngine&)>;
  struct Generator {
    MultiplicityGenerator multiplicity = nullptr;
    VertexGenerator vertex = nullptr;
    ParticlesGenerator particles = nullptr;
  };

}// namespace ActsExamples
