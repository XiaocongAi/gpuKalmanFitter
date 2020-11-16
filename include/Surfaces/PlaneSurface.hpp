// This file is part of the Acts project.
//
// Copyright (C) 2016-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// PlaneSurface.h, Acts project
///////////////////////////////////////////////////////////////////

#pragma once

#include "Geometry/GeometryContext.hpp"
#include "Geometry/GeometryStatics.hpp"

#include "Material/HomogeneousSurfaceMaterial.hpp"
#include "Surfaces/InfiniteBounds.hpp"
#include "Surfaces/PlanarBounds.hpp"
#include "Surfaces/Surface.hpp"
#include "Surfaces/detail/PlanarHelper.hpp"

#include "Utilities/Definitions.hpp"

#include <limits>

namespace Acts {

/// @class PlaneSurface
///
/// Class for a planaer in the TrackingGeometry.
///
/// The PlaneSurface extends the Surface class with the possibility to
/// convert local to global positions (vice versa).
///
/// @image html PlaneSurface.png
///
template <typename surface_bounds_t = InfiniteBounds>
class PlaneSurface : public Surface {
  friend Surface;

public:
  using SurfaceBoundsType = surface_bounds_t;

  //// Default Constructor
  PlaneSurface() = default;

  /// Copy Constructor
  ///
  /// @param psf is the source surface for the copy
  // @note The ACTS_DEVICE_FUNC identifier is necessary, otherwise CUDA
  // complaint
  ACTS_DEVICE_FUNC PlaneSurface(const PlaneSurface &other);

  /// Copy constructor - with shift
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param other is the source cone surface
  /// @param transf is the additional transfrom applied after copying
  PlaneSurface(const GeometryContext &gctx, const PlaneSurface &other,
               const Transform3D &transf);

  /// Dedicated Constructor with normal vector
  /// This is for curvilinear surfaces which are by definition boundless
  ///
  /// @param center is the center position of the surface
  /// @param normal is the normal vector of the plane surface
  ACTS_DEVICE_FUNC PlaneSurface(const Vector3D &center, const Vector3D &normal);

  /// Dedicated Constructor with normal vector
  /// This is for curvilinear surfaces which are by definition boundless
  ///
  /// @param center is the center position of the surface
  /// @param normal is the normal vector of the plane surface
  /// @param material is the surface material
  ACTS_DEVICE_FUNC PlaneSurface(const Vector3D &center, const Vector3D &normal,
                                const HomogeneousSurfaceMaterial &material);

  /// Constructor for Planes with bounds object
  ///
  /// @param htrans transform in 3D that positions this surface
  /// @param pbounds bounds object to describe the actual surface area
  template <
      typename T = surface_bounds_t,
      std::enable_if_t<not std::is_same<T, InfiniteBounds>::value, int> = 0>
  PlaneSurface(const Transform3D &htrans, const surface_bounds_t *pbounds);

public:
  /// Destructor - defaulted
  ~PlaneSurface() = default;

  /// Assignment operator
  ///
  /// @param other The source PlaneSurface for assignment
  ACTS_DEVICE_FUNC PlaneSurface &operator=(const PlaneSurface &other);

  /// Normal vector return
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param lposition is the local position is ignored
  ///
  /// return a Vector3D by value
  ACTS_DEVICE_FUNC const Vector3D normal(const GeometryContext &gctx,
                                         const Vector2D &lposition) const;

  /// The binning position is the position calcualted
  /// for a certain binning type
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param bValue is the binning type to be used
  ///
  /// @return position that can beused for this binning
  const Vector3D binningPosition(const GeometryContext &gctx,
                                 BinningValue bValue) const final;

  /// Return the surface type
  ACTS_DEVICE_FUNC SurfaceType type() const;

  /// Return method for bounds object of this surfrace
  ACTS_DEVICE_FUNC const surface_bounds_t *bounds() const;

  /// Return mehtod for the reference frame
  /// This is the frame in which the covariance matrix is defined (specialized
  /// by all surfaces)
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position global 3D position - considered to be on surface but not
  /// inside bounds (check is done)
  /// @param momentum global 3D momentum representation (optionally ignored)
  ///
  /// @return RotationMatrix3D which defines the three axes of the measurement
  /// frame
  ACTS_DEVICE_FUNC const Acts::RotationMatrix3D
  referenceFrame(const GeometryContext &gctx, const Vector3D &position,
                 const Vector3D &momentum) const;

  /// Initialize the jacobian from local to global
  /// the surface knows best, hence the calculation is done here.
  /// The jacobian is assumed to be initialised, so only the
  /// relevant entries are filled
  ///
  /// @todo this mixes track parameterisation and geometry
  /// should move to :
  /// "Acts/EventData/detail/coordinate_transformations.hpp"
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param jacobian is the jacobian to be initialized
  /// @param position is the global position of the parameters
  /// @param direction is the direction at of the parameters
  /// @param pars is the parameter vector
  ACTS_DEVICE_FUNC void initJacobianToGlobal(const GeometryContext &gctx,
                                             BoundToFreeMatrix &jacobian,
                                             const Vector3D &position,
                                             const Vector3D &direction,
                                             const BoundVector &pars) const;

  /// Initialize the jacobian from global to local
  /// the surface knows best, hence the calculation is done here.
  /// The jacobian is assumed to be initialised, so only the
  /// relevant entries are filled
  ///
  /// @param jacobian is the jacobian to be initialized
  /// @param position is the global position of the parameters
  /// @param direction is the direction at of the parameters
  /// @param gctx The current geometry context object, e.g. alignment
  ///
  /// @return the transposed reference frame (avoids recalculation)
  ACTS_DEVICE_FUNC const RotationMatrix3D initJacobianToLocal(
      const GeometryContext &gctx, FreeToBoundMatrix &jacobian,
      const Vector3D &position, const Vector3D &direction) const;

  /// Calculate the form factors for the derivatives
  /// the calculation is identical for all surfaces where the
  /// reference frame does not depend on the direction
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position is the position of the paramters in global
  /// @param direction is the direction of the track
  /// @param rft is the transposed reference frame (avoids recalculation)
  /// @param jacobian is the transport jacobian
  ///
  /// @return a five-dim vector
  ACTS_DEVICE_FUNC const BoundRowVector
  derivativeFactors(const GeometryContext &gctx, const Vector3D &position,
                    const Vector3D &direction, const RotationMatrix3D &rft,
                    const BoundToFreeMatrix &jacobian) const;

  /// Local to global transformation
  /// For planar surfaces the momentum is ignroed in the local to global
  /// transformation
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param lposition local 2D position in specialized surface frame
  /// @param momentum global 3D momentum representation (optionally ignored)
  /// @param position global 3D position to be filled (given by reference for
  /// method symmetry)
  ACTS_DEVICE_FUNC void localToGlobal(const GeometryContext &gctx,
                                      const Vector2D &lposition,
                                      const Vector3D &momentum,
                                      Vector3D &position) const;

  /// Global to local transformation
  /// For planar surfaces the momentum is ignroed in the global to local
  /// transformation
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position global 3D position - considered to be on surface but not
  /// inside bounds (check is done)
  /// @param momentum global 3D momentum representation (optionally ignored)
  /// @param lposition local 2D position to be filled (given by reference for
  /// method symmetry)
  ///
  /// @return boolean indication if operation was successful (fail means global
  /// position was not on surface)
  ACTS_DEVICE_FUNC bool globalToLocal(const GeometryContext &gctx,
                                      const Vector3D &position,
                                      const Vector3D &momentum,
                                      Vector2D &lposition) const;

  /// Method that calculates the correction due to incident angle
  ///
  /// @param position global 3D position - considered to be on surface but not
  /// inside bounds (check is done)
  /// @param direction global 3D momentum direction (ignored for PlaneSurface)
  /// @note this is the final implementation of the pathCorrection function
  ///
  /// @return a double representing the scaling factor
  ACTS_DEVICE_FUNC double pathCorrection(const GeometryContext &gctx,
                                         const Vector3D &position,
                                         const Vector3D &direction) const;

  /// Straight line intersection schema from position/direction
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position The position to start from
  /// @param direction The direction at start
  /// @param bcheck the Boundary Check
  ///
  /// @return SurfaceIntersection object (contains intersection & surface)
  ACTS_DEVICE_FUNC SurfaceIntersection
  intersect(const GeometryContext &gctx, const Vector3D &position,
            const Vector3D &direction, const BoundaryCheck &bcheck) const;

  /// Return properly formatted class name for screen output
  // std::string name() const override;

protected:
  /// the bounds of this surface
  const surface_bounds_t *m_bounds = nullptr;
};

#include "Surfaces/detail/PlaneSurface.ipp"

} // end of namespace Acts
