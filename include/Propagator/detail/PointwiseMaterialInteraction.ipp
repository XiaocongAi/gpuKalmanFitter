// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Material/Interactions.hpp"
#include "Utilities/Helpers.hpp"

namespace Acts {
namespace detail {
inline void PointwiseMaterialInteraction::evaluatePointwiseMaterialInteraction(
    bool multipleScattering, bool energyLoss) {
  if (energyLoss) {
    Eloss = computeEnergyLossBethe(slab, pdg, mass, qOverP, q);
  }
  // Compute contributions from interactions
  if (performCovarianceTransport) {
    covarianceContributions(multipleScattering, energyLoss);
  }
}

inline void
PointwiseMaterialInteraction::covarianceContributions(bool multipleScattering,
                                                      bool energyLoss) {
  // Compute contributions from interactions
  if (multipleScattering) {
    // TODO use momentum before or after energy loss in backward mode?
    const auto theta0 =
        computeMultipleScatteringTheta0(slab, pdg, mass, qOverP, q);
    // sigmaPhi = theta0 / sin(theta)
    const auto sigmaPhi = theta0 * (dir.norm() / VectorHelpers::perp(dir));
    variancePhi = sigmaPhi * sigmaPhi;
    // sigmaTheta = theta0
    varianceTheta = theta0 * theta0;
  }
  // TODO just ionisation loss or full energy loss?
  if (energyLoss) {
    const auto sigmaQoverP =
        computeEnergyLossLandauSigmaQOverP(slab, pdg, mass, qOverP, q);
    varianceQoverP = sigmaQoverP * sigmaQoverP;
  }
}

inline ActsScalar PointwiseMaterialInteraction::updateVariance(
    ActsScalar variance, ActsScalar change, NoiseUpdateMode updateMode) const {
  // Add/Subtract the change
  // Protect the variance against becoming negative
  return std::max(static_cast<ActsScalar>(0.),
                  variance + std::copysign(change, updateMode));
}
} // namespace detail
} // end of namespace Acts
