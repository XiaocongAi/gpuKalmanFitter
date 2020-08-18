#pragma once

#include "EventData/PixelSourceLink.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Units.hpp"
#include <random>

std::default_random_engine generator(42);
std::normal_distribution<double> gauss(0., 1.);

using namespace Acts;

// Struct for B field
struct ConstantBField {
  ACTS_DEVICE_FUNC static Vector3D getField(const Vector3D & /*field*/) {
    return Vector3D(0., 0., 2. * Acts::units::_T);
  }
};

// Measurement creator
struct MeasurementCreator {
  double resX = 30 * Acts::units::_um;
  double resY = 30 * Acts::units::_um;

  struct this_result {
    std::vector<PixelSourceLink> sourcelinks;
  };
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t &state, const stepper_t &stepper,
                  result_type &result) const {
    if (state.navigation.currentSurface != nullptr) {
      // Apply global to local
      Vector2D lPos;
      state.navigation.currentSurface->globalToLocal(
          state.options.geoContext, stepper.position(state.stepping),
          stepper.direction(state.stepping), lPos);
      // Perform the smearing to truth
      double dx = resX * gauss(generator);
      double dy = resY * gauss(generator);

      // The measurement values
      Vector2D values;
      values << lPos[0] + dx, lPos[1] + dy;

      // The measurement covariance
      SymMatrix2D cov;
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
