#pragma once

#include "EventData/PixelSourceLink.hpp"
#include "EventData/TrackParameters.hpp"
#include "Material/Material.hpp"
#include "Material/MaterialSlab.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Units.hpp"

#include <fstream>
#include <iostream>
#include <random>

namespace Test {

// Struct for B field
struct ConstantBField {
  ACTS_DEVICE_FUNC static Acts::Vector3D
  getField(const Acts::Vector3D & /*field*/) {
    return Acts::Vector3D(0., 0., 2. * Acts::units::_T);
  }
};

// Silicon material
Acts::Material makeSilicon() {
  return Acts::Material::fromMolarDensity(
      9.370 * Acts::units::_cm, 46.52 * Acts::units::_cm, 28.0855, 14,
      (2.329 / 28.0855) * Acts::UnitConstants::mol / Acts::UnitConstants::cm3);
}

// Measurement creator
template <typename generator_t> struct MeasurementCreator {
  /// Random number generator used for the simulation.
  generator_t *generator = nullptr;

  // The smearing resolution
  ActsScalar resX = 30 * Acts::units::_um;
  ActsScalar resY = 30 * Acts::units::_um;

  struct this_result {
    std::vector<Acts::PixelSourceLink> sourcelinks;
  };
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t &state, const stepper_t &stepper,
                  result_type &result) const {
    assert(generator and "The generator pointer must be valid");

    if (state.navigation.currentSurface != nullptr) {
      // Apply global to local
      Acts::Vector2D lPos;
      state.navigation.currentSurface
          ->globalToLocal<propagator_state_t::NavigationSurface>(
              state.options.geoContext, stepper.position(state.stepping),
              stepper.direction(state.stepping), lPos);
      // Perform the smearing to truth
      ActsScalar dx =
          std::normal_distribution<ActsScalar>(0., resX)(*generator);
      ActsScalar dy =
          std::normal_distribution<ActsScalar>(0., resY)(*generator);

      // The measurement values
      Acts::Vector2D values;
      values << lPos[0] + dx, lPos[1] + dy;

      // The measurement covariance
      Acts::SymMatrix2D cov;
      cov << resX * resX, 0., 0., resY * resY;

      // Push back to the container
      result.sourcelinks.emplace_back(values, cov,
                                      state.navigation.currentSurface);
    }
    return;
  }
};

// Test actor
struct VoidActor {
  struct this_result {
    bool status = false;
  };
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  ACTS_DEVICE_FUNC void operator()(propagator_state_t &state,
                                   const stepper_t &stepper,
                                   result_type &result) const {
    return;
  }
};

// Test aborter
struct VoidAborter {
  template <typename propagator_state_t, typename stepper_t, typename result_t>
  ACTS_DEVICE_FUNC bool operator()(propagator_state_t &state,
                                   const stepper_t &stepper,
                                   result_t &result) const {
    return false;
  }
};

} // namespace Test
