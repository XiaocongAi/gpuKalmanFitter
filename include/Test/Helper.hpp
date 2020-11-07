#pragma once

#include "EventData/PixelSourceLink.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Units.hpp"
#include <random>

namespace Test{

// Struct for B field
struct ConstantBField {
  ACTS_DEVICE_FUNC static Vector3D getField(const Vector3D & /*field*/) {
    return Vector3D(0., 0., 2. * Acts::units::_T);
  }
};

// Measurement creator
template< typename generator_t>
struct MeasurementCreator {
  /// Random number generator used for the simulation.
  generator_t *generator = nullptr;
 
  // The smearing resolution 
  double resX = 30 * Acts::units::_um;
  double resY = 30 * Acts::units::_um;

  struct this_result {
    std::vector<PixelSourceLink> sourcelinks;
  };
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t &state, const stepper_t &stepper,
                  result_type &result) const {
    assert(generator and "The generator pointer must be valid"); 

    if (state.navigation.currentSurface != nullptr) {
      // Apply global to local
      Acts::Vector2D lPos;
      state.navigation.currentSurface->globalToLocal(
          state.options.geoContext, stepper.position(state.stepping),
          stepper.direction(state.stepping), lPos);
      // Perform the smearing to truth
      double dx = std::normal_distribution<double>(0., resX)(*generator);
      double dy = std::normal_distribution<double>(0., resY)(*generator);

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
