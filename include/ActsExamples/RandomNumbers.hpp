// This file is part of the Acts project.
//
// Copyright (C) 2017 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//
//  RandomNumbers.hpp
//
//  Created by Andreas Salzburger on 17/05/16.
//
//

#pragma once

#include <cstdint>
#include <random>

namespace ActsExamples {

/// The random number generator used in the framework.
using RandomEngine = std::mt19937; ///< Mersenne Twister

/// Provide event and algorithm specific random number generator.s
///
/// This provides local random number generators, allowing for
/// thread-safe, lock-free, and reproducible random number generation across
/// single-threaded and multi-threaded test framework runs.
///
/// The role of the RandomNumbers is only to spawn local random number
/// generators. It does not, in and of itself, accomodate requests for specific
/// random number distributions (uniform, gaussian, etc). For this purpose,
/// clients should spawn their own local distribution objects
/// as needed, following the C++11 STL design.
class RandomNumbers {
public:
  struct Config {
    uint64_t seed = 1234567890u; ///< random seed
  };

  RandomNumbers(const Config &cfg);

  /// Spawn an event-local random number generator.
  ///
  /// It calls generateSeed() for an event driven seed
  ///
  RandomEngine spawnGenerator(uint64_t eventNumber) const;

  /// Generate a event and algorithm specific seed value.
  ///
  /// This should only be used in special cases e.g. where a custom
  /// random engine is used and `spawnGenerator` can not be used.
  uint64_t generateSeed(uint64_t eventNumber) const;

private:
  Config m_cfg;
};

inline RandomNumbers::RandomNumbers(const Config &cfg) : m_cfg(cfg) {}

inline RandomEngine RandomNumbers::spawnGenerator(uint64_t eventNumber) const {
  return RandomEngine(generateSeed(eventNumber));
}

inline uint64_t RandomNumbers::generateSeed(uint64_t eventNumber) const {
  const uint64_t k2 = eventNumber;
  const uint64_t id = k2 * (k2 + 1) / 2 + k2;
  return m_cfg.seed + id;
}

} // namespace ActsExamples
