#pragma once

#include "Utilities/Definitions.hpp"

namespace Acts {
struct TrackParameters {
  /// Default constructor
  TrackParameters() = default;

  /// Constructor from arguments
  ACTS_DEVICE_FUNC TrackParameters(const Vector3D &position,
                                   const Vector3D &momentum, int charge)
      : m_vPosition(position), m_vMomentum(momentum), m_charge(charge) {}

  /// @brief access position in global coordinate system
  ///
  /// @return 3D vector with global position
  ACTS_DEVICE_FUNC Vector3D position() const { return m_vPosition; }

  /// @brief access momentum in global coordinate system
  ///
  /// @return 3D vector with global momentum
  ACTS_DEVICE_FUNC Vector3D momentum() const { return m_vMomentum; }

  ACTS_DEVICE_FUNC int charge() const { return m_charge; }

private:
  Vector3D m_vPosition = {0, 0, 0}; ///< 3D vector with global position
  Vector3D m_vMomentum = {0, 0, 0}; ///< 3D vector with global momentum
  int m_charge = 1;                 ///< The charge
};
} // namespace Acts
