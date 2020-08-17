// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// Surface.h, Acts project
///////////////////////////////////////////////////////////////////

#pragma once

// All functions callable from CUDA code must be qualified with __device__
#ifdef __CUDACC__
#define ACTS_DEVICE_FUNC __host__ __device__
// We need cuda_runtime.h to ensure that that EIGEN_USING_STD_MATH macro
// works properly on the device side
#include <cuda_runtime.h>
#else
#define ACTS_DEVICE_FUNC 
#endif

#include "Geometry/GeometryContext.hpp"
#include "Geometry/GeometryObject.hpp"
#include "Geometry/GeometryStatics.hpp"
#include "Surfaces/BoundaryCheck.hpp"
#include "Surfaces/SurfaceBounds.hpp"
#include "Utilities/Definitions.hpp"
#include "Utilities/Intersection.hpp"
#include "Surfaces/detail/PlanarHelper.hpp"

#include <memory>

namespace Acts {

class Surface;
class SurfaceBounds;

/// Typedef of the surface intersection
using SurfaceIntersection = ObjectIntersection<Surface>;

/// @class Surface
///
/// @brief Abstract Base Class for tracking surfaces
///
/// The Surface class builds the core of the Acts Tracking Geometry.
/// All other geometrical objects are either extending the surface or
/// are built from it.
///
/// Surfaces are either owned by Detector elements or the Tracking Geometry,
/// in which case they are not copied within the data model objects.
///
class Surface : public virtual GeometryObject {
public:
  /// @enum SurfaceType
  ///
  /// This enumerator simplifies the persistency & calculations,
  /// by saving a dynamic_cast, e.g. for persistency
  enum SurfaceType {
    Cone = 0,
    Cylinder = 1,
    Disc = 2,
    Perigee = 3,
    Plane = 4,
    Straw = 5,
    Curvilinear = 6,
    Other = 7
  };

protected:
  /// Default constructor
  Surface() = default;

  /// Constructor with Transform3
  ///
  /// @param tform Transform3D positions the surface in 3D global space
  /// @note also acts as default constructor
  Surface(const Transform3D &tform);

  /// Copy constructor
  ///
  /// @note copy construction invalidates the association
  /// to detector element and layer
  ///
  /// @param other Source surface for copy.
  Surface(const Surface &other);

  /// Copy constructor with optional shift
  ///
  /// @note copy construction invalidates the association
  /// to detector element and layer
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param other Source surface for copy
  /// @param shift Additional transform applied after copying from the source
  Surface(const GeometryContext &gctx, const Surface &other,
                           const Transform3D &shift);

  /// Dedicated Constructor with normal vector
  /// This is for curvilinear surfaces which are by definition boundless
  ///
  /// @param center is the center position of the surface
  /// @param normal is thenormal vector of the plane surface
  ACTS_DEVICE_FUNC Surface(const Vector3D &center, const Vector3D &normal);

public:
  /// Destructor
  virtual ~Surface() = default;

  /// Assignment operator
  /// @note copy construction invalidates the association
  /// to detector element and layer
  ///
  /// @param other Source surface for the assignment
  ACTS_DEVICE_FUNC Surface &operator=(const Surface &other);

  /// Comparison (equality) operator
  /// The strategy for comparison is
  /// (a) first pointer comparison
  /// (b) then type comparison
  /// (c) then bounds comparison
  /// (d) then transform comparison
  ///
  /// @param other source surface for the comparison
  ACTS_DEVICE_FUNC bool operator==(const Surface &other) const;

  /// Comparison (non-equality) operator
  ///
  /// @param sf Source surface for the comparison
  ACTS_DEVICE_FUNC bool operator!=(const Surface &sf) const;

public:
  /// Return method for the Surface type to avoid dynamic casts
  virtual SurfaceType type() const = 0;

  /// Return method for the surface Transform3D by reference
  /// In case a detector element is associated the surface transform
  /// is just forwarded to the detector element in order to keep the
  /// (mis-)alignment cache cetrally handled
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  ///
  /// @return the contextual transform
  ACTS_DEVICE_FUNC const Transform3D &
  transform(const GeometryContext &gctx) const;

  /// Return method for the surface center by reference
  /// @note the center is always recalculated in order to not keep a cache
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  ///
  /// @return center position by value
  ACTS_DEVICE_FUNC 
	  const Vector3D
  center(const GeometryContext &gctx) const;

  /// Return method for the normal vector of the surface
  /// The normal vector can only be generally defined at a given local position
  /// It requires a local position to be given (in general)
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param lposition is the local position where the normal vector is
  /// constructed
  ///
  /// @return normal vector by value
  ACTS_DEVICE_FUNC const Vector3D
  normal(const GeometryContext &gctx, const Vector2D &lposition) const;

  /// Return method for the normal vector of the surface
  /// The normal vector can only be generally defined at a given local position
  /// It requires a local position to be given (in general)
  ///
  /// @param position is the global position where the normal vector is
  /// constructed
  /// @param gctx The current geometry context object, e.g. alignment

  ///
  /// @return normal vector by value
  ACTS_DEVICE_FUNC const Vector3D
  normal(const GeometryContext &gctx, const Vector3D &position) const;

  /// Return method for the normal vector of the surface
  ///
  /// It will return a normal vector at the center() position
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  //
  /// @return normal vector by value
  ACTS_DEVICE_FUNC const Vector3D
  normal(const GeometryContext &gctx) const {
    return normal(gctx, center(gctx));
  }

  /// Return method for SurfaceBounds
  /// @return SurfaceBounds by reference
  //ACTS_DEVICE_FUNC virtual const SurfaceBounds &bounds() const = 0;

  /// The geometric onSurface method
  ///
  /// Geometrical check whether position is on Surface
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position global position to be evaludated
  /// @param momentum global momentum (required for line-type surfaces)
  /// @param bcheck BoundaryCheck directive for this onSurface check
  ///
  /// @return boolean indication if operation was successful
  ACTS_DEVICE_FUNC bool isOnSurface(const GeometryContext &gctx,
                                    const Vector3D &position,
                                    const Vector3D &momentum,
                                    const BoundaryCheck &bcheck = true) const;

  /// The insideBounds method for local positions
  ///
  /// @param lposition The local position to check
  /// @param bcheck BoundaryCheck directive for this onSurface check
  /// @return boolean indication if operation was successful
  ACTS_DEVICE_FUNC  bool
  insideBounds(const Vector2D &lposition,
               const BoundaryCheck &bcheck = true) const;

  /// Local to global transformation
  /// Generalized local to global transformation for the surface types. Since
  /// some surface types need the global momentum/direction to resolve sign
  /// ambiguity this is also provided
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
  /// Generalized global to local transformation for the surface types. Since
  /// some surface types need the global momentum/direction to resolve sign
  /// ambiguity this is also provided
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
  ACTS_DEVICE_FUNC void
  initJacobianToGlobal(const GeometryContext &gctx, BoundToFreeMatrix &jacobian,
                       const Vector3D &position, const Vector3D &direction,
                       const BoundVector &pars) const;

  /// Initialize the jacobian from global to local
  /// the surface knows best, hence the calculation is done here.
  /// The jacobian is assumed to be initialised, so only the
  /// relevant entries are filled
  ///
  /// @todo this mixes track parameterisation and geometry
  /// should move to :
  /// "Acts/EventData/detail/coordinate_transformations.hpp"
  ///
  /// @param jacobian is the jacobian to be initialized
  /// @param position is the global position of the parameters
  /// @param direction is the direction at of the parameters
  /// @param gctx The current geometry context object, e.g. alignment
  ///
  /// @return the transposed reference frame (avoids recalculation)
  ACTS_DEVICE_FUNC const RotationMatrix3D
  initJacobianToLocal(const GeometryContext &gctx, FreeToBoundMatrix &jacobian,
                      const Vector3D &position,
                      const Vector3D &direction) const;

  /// Calculate the form factors for the derivatives
  /// the calculation is identical for all surfaces where the
  /// reference frame does not depend on the direction
  ///
  ///
  /// @todo this mixes track parameterisation and geometry
  /// should move to :
  /// "Acts/EventData/detail/coordinate_transformations.hpp"
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

  /// Calucation of the path correction for incident
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position global 3D position - considered to be on surface but not
  /// inside bounds (check is done)
  /// @param direction global 3D momentum direction
  ///
  /// @return Path correction with respect to the nominal incident.
  ACTS_DEVICE_FUNC double
  pathCorrection(const GeometryContext &gctx, const Vector3D &position,
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
            const Vector3D &direction, const BoundaryCheck &bcheck) const

  {
    // Get the intersection with the surface
    Intersection sIntersection =
        intersectionEstimate(gctx, position, direction, bcheck);
    // return a surface intersection with result direction
    return SurfaceIntersection(sIntersection, this);
  }

  /// Straight line intersection from position and momentum
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position global 3D position - considered to be on surface but not
  ///        inside bounds (check is done)
  /// @param direction 3D direction representation - expected to be normalized
  ///        (no check done)
  /// @param bcheck boundary check directive for this operation
  ///
  /// @return Intersection object
  ACTS_DEVICE_FUNC Intersection
  intersectionEstimate(const GeometryContext &gctx, const Vector3D &position,
                       const Vector3D &direction,
                       const BoundaryCheck &bcheck) const;

  /// Return properly formatted class name
  //virtual std::string name() const = 0;

protected:
  /// Transform3D definition that positions
  /// (translation, rotation) the surface in global space
  //Transform3D m_transform = s_idTransform;
  Transform3D m_transform = Transform3D::Identity();
};

#include "Surfaces/detail/Surface.ipp"

} // namespace Acts
