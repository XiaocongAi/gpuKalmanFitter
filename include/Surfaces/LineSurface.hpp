// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once
#include "Geometry/GeometryContext.hpp"
#include "Geometry/GeometryStatics.hpp"
#include "Surfaces/InfiniteBounds.hpp"
#include "Surfaces/Surface.hpp"
#include "Utilities/Definitions.hpp"

namespace Acts {

///  @class LineSurface without infinite bounds
///
///  Base class for a linear surfaces in the TrackingGeometry
///  to describe dirft tube, straw like detectors or the Perigee
///  It inherits from Surface.
///
///  @note It leaves the type() method virtual, so it can not be instantiated
///
/// @image html LineSurface.png
class LineSurface : public Surface {
  friend Surface;

public:
  using SurfaceBoundsType = InfiniteBounds;

  //// Default Constructor
  LineSurface() = default;

  /// Copy constructor
  ///
  /// @param other The source surface for copying
  ACTS_DEVICE_FUNC LineSurface(const LineSurface &other);

  /// Copy constructor - with shift
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param other is the source cone surface
  /// @param shift is the additional transform applied after copying
  LineSurface(const GeometryContext &gctx, const LineSurface &other,
              const Transform3D &shift);

  /// Constructor from the transform
  ///
  LineSurface(const Transform3D &transform);

  /// Constructor from GlobalPosition
  ///
  /// @note Constructor taken from Acts::PerigeeSurface
  /// @param gp position where the perigee is centered
  ACTS_DEVICE_FUNC LineSurface(const Vector3D &gp);

public:
  ~LineSurface() = default;

  /// Assignment operator
  ///
  /// @param slsf is the source surface dor copying
  ACTS_DEVICE_FUNC LineSurface &operator=(const LineSurface &other);

  /// Normal vector return
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param lposition is the local position is ignored
  ///
  /// @return a Vector3D by value
  ACTS_DEVICE_FUNC Vector3D normal(const GeometryContext &gctx,
                                   const Vector2D &lposition) const;

  /// Normal vector return without argument
  using Surface::normal;

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

  /// This method returns the bounds of the Surface by reference */
  ACTS_DEVICE_FUNC const InfiniteBounds *bounds() const;

  /// Return the measurement frame - this is needed for alignment, in particular
  ///
  /// for StraightLine and Perigee Surface
  ///  - the default implementation is the the RotationMatrix3D of the transform
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position is the global position where the measurement frame is
  /// constructed
  /// @param momentum is the momentum used for the measurement frame
  /// construction
  ///
  /// @return is a rotation matrix that indicates the measurement frame
  ACTS_DEVICE_FUNC const RotationMatrix3D
  referenceFrame(const GeometryContext &gctx, const Vector3D &position,
                 const Vector3D &momentum) const;

  /// Initialize the jacobian from local to global
  /// the surface knows best, hence the calculation is done here.
  /// The jacobian is assumed to be initialised, so only the
  /// relevant entries are filled
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param jacobian is the jacobian to be initialized
  /// @param position is the global position of the parameters
  /// @param direction is the direction at of the parameters
  ///
  /// @param pars is the paranmeters vector
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
  /// @note This is the same with the PlaneSurface. But without using the
  /// virtual function, we have duplicate the code here
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
  /// for line surfaces the momentum is used in order to interpret the drift
  /// radius
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param lposition is the local position to be transformed
  /// @param momentum is the global momentum (used to sign the closest approach)
  /// @param position is the global position which is filled
  ACTS_DEVICE_FUNC void localToGlobal(const GeometryContext &gctx,
                                      const Vector2D &lposition,
                                      const Vector3D &momentum,
                                      Vector3D &position) const;

  /// Specified for LineSurface: global to local method without dynamic
  /// memory allocation
  ///
  /// This method is the true global->local transformation.<br>
  /// makes use of globalToLocal and indicates the sign of the Acts::eBoundLoc0
  /// by the given momentum
  ///
  /// The calculation of the sign of the radius (or \f$ d_0 \f$) can be done as
  /// follows:<br>
  /// May \f$ \vec d = \vec m - \vec c \f$ denote the difference between the
  /// center of the line and the global position of the measurement/predicted
  /// state,
  /// then \f$ \vec d
  /// \f$
  /// lies within the so
  /// called measurement plane.
  /// The measurement plane is determined by the two orthogonal vectors \f$
  /// \vec{measY}= \vec{Acts::eBoundLoc1} \f$
  /// and \f$ \vec{measX} = \vec{measY} \times \frac{\vec{p}}{|\vec{p}|}
  /// \f$.<br>
  ///
  /// The sign of the radius (\f$ d_{0} \f$ ) is then defined by the projection
  /// of
  /// \f$ \vec{d} \f$
  /// onto \f$ \vec{measX} \f$:<br>
  /// \f$ sign = -sign(\vec{d} \cdot \vec{measX}) \f$
  ///
  /// \image html SignOfDriftCircleD0.gif
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

  /// the pathCorrection for derived classes with thickness
  /// is by definition 1 for LineSurfaces
  ///
  /// @note input parameters are ignored
  /// @note there's no material associated to the line surface
  ACTS_DEVICE_FUNC ActsScalar pathCorrection(const GeometryContext &gctx,
                                             const Vector3D &position,
                                             const Vector3D &momentum) const;

  /// @brief Straight line intersection schema
  ///
  /// @param gctx The current geometry context object, e.g. alignment
  /// @param position The global position as a starting point
  /// @param direction The global direction at the starting point
  ///        @note exptected to be normalized
  /// @param bcheck The boundary check directive for the estimate
  ///
  ///   <b>mathematical motivation:</b>
  ///   Given two lines in parameteric form:<br>
  ///   - @f$ \vec l_{a}(\lambda) = \vec m_a + \lambda \cdot \vec e_{a} @f$ <br>
  ///   - @f$ \vec l_{b}(\mu) = \vec m_b + \mu \cdot \vec e_{b} @f$ <br>
  ///   the vector between any two points on the two lines is given by:
  ///   - @f$ \vec s(\lambda, \mu) = \vec l_{b} - l_{a} = \vec m_{ab} + \mu
  ///   \cdot
  ///  \vec e_{b} - \lambda \cdot \vec e_{a} @f$, <br>
  ///   when @f$ \vec m_{ab} = \vec m_{b} - \vec m_{a} @f$.<br>
  ///   @f$ \vec s(u, \mu_0) @f$  denotes the vector between the two
  ///  closest points <br>
  ///   @f$ \vec l_{a,0} = l_{a}(u) @f$ and @f$ \vec l_{b,0} =
  ///  l_{b}(\mu_0) @f$ <br>
  ///   and is perpendicular to both, @f$ \vec e_{a} @f$ and @f$ \vec e_{b} @f$.
  ///
  ///   This results in a system of two linear equations:<br>
  ///   - (i) @f$ 0 = \vec s(u, \mu_0) \cdot \vec e_a = \vec m_ab \cdot
  ///  \vec e_a + \mu_0 \vec e_a \cdot \vec e_b - u @f$ <br>
  ///   - (ii) @f$ 0 = \vec s(u, \mu_0) \cdot \vec e_b = \vec m_ab \cdot
  ///  \vec e_b + \mu_0  - u \vec e_b \cdot \vec e_a @f$ <br>
  ///
  ///   Solving (i), (ii) for @f$ u @f$ and @f$ \mu_0 @f$ yields:
  ///   - @f$ u = \frac{(\vec m_ab \cdot \vec e_a)-(\vec m_ab \cdot \vec
  ///  e_b)(\vec e_a \cdot \vec e_b)}{1-(\vec e_a \cdot \vec e_b)^2} @f$ <br>
  ///   - @f$ \mu_0 = - \frac{(\vec m_ab \cdot \vec e_b)-(\vec m_ab \cdot \vec
  ///  e_a)(\vec e_a \cdot \vec e_b)}{1-(\vec e_a \cdot \vec e_b)^2} @f$ <br>
  ///
  /// @return is the intersection object
  ACTS_DEVICE_FUNC SurfaceIntersection intersect(
      const GeometryContext &gctx, const Vector3D &position,
      const Vector3D &direction, const BoundaryCheck &bcheck = false) const;

  /// Return properly formatted class name for screen output */
  // std::string name() const override;
};

#include "Surfaces/detail/LineSurface.ipp"

} // namespace Acts
