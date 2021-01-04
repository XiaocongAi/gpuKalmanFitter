// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Utilities/IAxis.hpp"
#include "Utilities/detail/Axis.hpp"
#include <array>
#include <set>
#include <tuple>
#include <utility>

namespace Acts {

namespace detail {

// This object can be iterated to produce the (ordered) set of global indices
// associated with a neighborhood around a certain point on a grid.
//
// The goal is to emulate the effect of enumerating the global indices into
// an std::set (or into an ActsVectorX that gets subsequently sorted), without
// paying the price of dynamic memory allocation in hot magnetic field
// interpolation code.
//
template <size_t DIM> class GlobalNeighborHoodIndices {
public:
  // You can get the local neighbor indices from
  // grid_helper_impl<DIM>::neighborHoodIndices and the number of bins in
  // each direction from grid_helper_impl<DIM>::getNBins.
  ACTS_DEVICE_FUNC GlobalNeighborHoodIndices(
      ActsVector<NeighborHoodIndices, DIM> &neighborIndices,
      const ActsVector<size_t, DIM> &nBinsArray)
      : m_localIndices(neighborIndices) {
    if (DIM == 1)
      return;
    size_t globalStride = 1;
    for (long i = DIM - 2; i >= 0; --i) {
      globalStride *= (nBinsArray(i + 1) + 2);
      m_globalStrides(i) = globalStride;
    }
  }

  class iterator {
  public:
    iterator() = default;

    ACTS_DEVICE_FUNC
    iterator(const GlobalNeighborHoodIndices &parent,
             ActsVector<NeighborHoodIndices::iterator, DIM> &&localIndicesIter)
        : m_localIndicesIter(std::move(localIndicesIter)), m_parent(&parent) {}

    ACTS_DEVICE_FUNC size_t operator*() const {
      size_t globalIndex = *m_localIndicesIter(DIM - 1);
      if (DIM == 1)
        return globalIndex;
      for (size_t i = 0; i < DIM - 1; ++i) {
        globalIndex += m_parent->m_globalStrides(i) * (*m_localIndicesIter(i));
      }
      return globalIndex;
    }

    ACTS_DEVICE_FUNC iterator &operator++() {
      const auto &localIndices = m_parent->m_localIndices;

      // Go to the next global index via a lexicographic increment:
      // - Start by incrementing the last local index
      // - If it reaches the end, reset it and increment the previous one...
      for (long i = DIM - 1; i > 0; --i) {
        ++m_localIndicesIter(i);
        if (m_localIndicesIter(i) != localIndices(i).end())
          return *this;
        m_localIndicesIter(i) = localIndices(i).begin();
      }

      // The first index should stay at the end value when it reaches it, so
      // that we know when we've reached the end of iteration.
      ++m_localIndicesIter(0);
      return *this;
    }

    ACTS_DEVICE_FUNC bool operator==(const iterator &it) {
      // We know when we've reached the end, so we don't need an end-iterator.
      // Sadly, in C++, there has to be one. Therefore, we special-case it
      // heavily so that it's super-efficient to create and compare to.
      if (it.m_parent == nullptr) {
        return m_localIndicesIter(0) == m_parent->m_localIndices(0).end();
      } else {
        bool isSame = true;
        for (size_t i = 0; i < DIM; i++) {
          if (m_localIndicesIter(i) != it.m_localIndicesIter(i)) {
            isSame = false;
            break;
          }
        }
        return isSame;
      }
    }

    ACTS_DEVICE_FUNC bool operator!=(const iterator &it) {
      return !(*this == it);
    }

  private:
    ActsVector<NeighborHoodIndices::iterator, DIM> m_localIndicesIter;
    const GlobalNeighborHoodIndices *m_parent = nullptr;
  };

  ACTS_DEVICE_FUNC iterator begin() const {
    ActsVector<NeighborHoodIndices::iterator, DIM> localIndicesIter;
    for (size_t i = 0; i < DIM; ++i) {
      localIndicesIter(i) = m_localIndices(i).begin();
    }
    return iterator(*this, std::move(localIndicesIter));
  }

  ACTS_DEVICE_FUNC iterator end() const { return iterator(); }

  // Number of indices that will be produced if this sequence is iterated
  ACTS_DEVICE_FUNC size_t size() const {
    size_t result = m_localIndices(0).size();
    for (size_t i = 1; i < DIM; ++i) {
      result *= m_localIndices(i).size();
    }
    return result;
  }

  // Collect the sequence of indices into an ActsVectorX
  ACTS_DEVICE_FUNC ActsVectorX<size_t> collect() const {
    ActsVectorX<size_t> result;
    result.resize(this->size());
    size_t iRow = 0;
    for (size_t idx : *this) {
      result(iRow) = idx;
      iRow++;
    }
    return result;
  }

private:
  ActsVector<NeighborHoodIndices, DIM> m_localIndices;
  ActsVector<size_t, DIM - 1> m_globalStrides;
};

/// @cond
/// @brief helper struct to calculate number of bins inside a grid
///
/// @tparam N number of axes to consider
template <size_t N> struct grid_helper_impl;

template <size_t N> struct grid_helper_impl {
  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getBinCenter(ActsVector<ActsScalar, sizeof...(Axes)> &center,
               const ActsVector<size_t, sizeof...(Axes)> &localIndices,
               const std::tuple<Axes...> &axes) {
    center(N) = std::get<N>(axes).getBinCenter(localIndices(N));
    grid_helper_impl<N - 1>::getBinCenter(center, localIndices, axes);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getGlobalBin(const ActsVector<size_t, sizeof...(Axes)> &localBins,
               const std::tuple<Axes...> &axes, size_t &bin, size_t &area) {
    const auto &thisAxis = std::get<N>(axes);
    bin += area * localBins(N);
    // make sure to account for under-/overflow bins
    area *= (thisAxis.getNBins() + 2);
    grid_helper_impl<N - 1>::getGlobalBin(localBins, axes, bin, area);
  }

  template <class Point, class... Axes>
  ACTS_DEVICE_FUNC static void
  getLocalBinIndices(const Point &point, const std::tuple<Axes...> &axes,
                     ActsVector<size_t, sizeof...(Axes)> &indices) {
    const auto &thisAxis = std::get<N>(axes);
    indices(N) = thisAxis.getBin(point[N]);
    grid_helper_impl<N - 1>::getLocalBinIndices(point, axes, indices);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getLocalBinIndices(size_t &bin, const std::tuple<Axes...> &axes, size_t &area,
                     ActsVector<size_t, sizeof...(Axes)> &indices) {
    const auto &thisAxis = std::get<N>(axes);
    // make sure to account for under-/overflow bins
    size_t new_area = area * (thisAxis.getNBins() + 2);
    grid_helper_impl<N - 1>::getLocalBinIndices(bin, axes, new_area, indices);
    indices(N) = bin / area;
    bin %= area;
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getLowerLeftBinEdge(ActsVector<ActsScalar, sizeof...(Axes)> &llEdge,
                      const ActsVector<size_t, sizeof...(Axes)> &localIndices,
                      const std::tuple<Axes...> &axes) {
    llEdge(N) = std::get<N>(axes).getBinLowerBound(localIndices(N));
    grid_helper_impl<N - 1>::getLowerLeftBinEdge(llEdge, localIndices, axes);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getLowerLeftBinIndices(ActsVector<size_t, sizeof...(Axes)> &localIndices,
                         const std::tuple<Axes...> &axes) {
    localIndices(N) = std::get<N>(axes).wrapBin(localIndices(N) - 1);
    grid_helper_impl<N - 1>::getLowerLeftBinIndices(localIndices, axes);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getNBins(const std::tuple<Axes...> &axes,
           ActsVector<size_t, sizeof...(Axes)> &nBinsArray) {
    // by convention getNBins does not include under-/overflow bins
    nBinsArray(N) = std::get<N>(axes).getNBins();
    grid_helper_impl<N - 1>::getNBins(axes, nBinsArray);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getAxes(const std::tuple<Axes...> &axes,
          ActsVector<const IAxis *, sizeof...(Axes)> &axesArr) {
    axesArr(N) = static_cast<const IAxis *>(&std::get<N>(axes));
    grid_helper_impl<N - 1>::getAxes(axes, axesArr);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getUpperRightBinEdge(ActsVector<ActsScalar, sizeof...(Axes)> &urEdge,
                       const ActsVector<size_t, sizeof...(Axes)> &localIndices,
                       const std::tuple<Axes...> &axes) {
    urEdge(N) = std::get<N>(axes).getBinUpperBound(localIndices(N));
    grid_helper_impl<N - 1>::getUpperRightBinEdge(urEdge, localIndices, axes);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getUpperRightBinIndices(ActsVector<size_t, sizeof...(Axes)> &localIndices,
                          const std::tuple<Axes...> &axes) {
    localIndices(N) = std::get<N>(axes).wrapBin(localIndices(N) + 1);
    grid_helper_impl<N - 1>::getUpperRightBinIndices(localIndices, axes);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getMin(const std::tuple<Axes...> &axes,
         ActsVector<ActsScalar, sizeof...(Axes)> &minArray) {
    minArray(N) = std::get<N>(axes).getMin();
    grid_helper_impl<N - 1>::getMin(axes, minArray);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getMax(const std::tuple<Axes...> &axes,
         ActsVector<ActsScalar, sizeof...(Axes)> &maxArray) {
    maxArray(N) = std::get<N>(axes).getMax();
    grid_helper_impl<N - 1>::getMax(axes, maxArray);
  }

  template <class Point, class... Axes>
  ACTS_DEVICE_FUNC static bool isInside(const Point &position,
                                        const std::tuple<Axes...> &axes) {
    bool insideThisAxis = std::get<N>(axes).isInside(position[N]);
    return insideThisAxis && grid_helper_impl<N - 1>::isInside(position, axes);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void neighborHoodIndices(
      const ActsVector<size_t, sizeof...(Axes)> &localIndices,
      std::pair<size_t, size_t> sizes, const std::tuple<Axes...> &axes,
      ActsVector<NeighborHoodIndices, sizeof...(Axes)> &neighborIndices) {
    // ask n-th axis
    size_t locIdx = localIndices(N);
    NeighborHoodIndices locNeighbors =
        std::get<N>(axes).neighborHoodIndices(locIdx, sizes);
    neighborIndices(N) = locNeighbors;

    grid_helper_impl<N - 1>::neighborHoodIndices(localIndices, sizes, axes,
                                                 neighborIndices);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  exteriorBinIndices(ActsVector<size_t, sizeof...(Axes)> &idx,
                     ActsVector<int, sizeof...(Axes)> isExterior,
                     std::set<size_t> &combinations,
                     const std::tuple<Axes...> &axes) {
    // iterate over this axis' bins, remembering which bins are exterior
    for (size_t i = 0; i < std::get<N>(axes).getNBins() + 2; ++i) {
      idx(N) = i;
      isExterior(N) = 0;
      if ((i == 0) || (i == std::get<N>(axes).getNBins() + 1)) {
        isExterior(N) = 1;
      }
      // vary other axes recursively
      grid_helper_impl<N - 1>::exteriorBinIndices(idx, isExterior, combinations,
                                                  axes);
    }
  }
};

template <> struct grid_helper_impl<0u> {
  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getBinCenter(ActsVector<ActsScalar, sizeof...(Axes)> &center,
               const ActsVector<size_t, sizeof...(Axes)> &localIndices,
               const std::tuple<Axes...> &axes) {
    center(0u) = std::get<0u>(axes).getBinCenter(localIndices(0u));
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getGlobalBin(const ActsVector<size_t, sizeof...(Axes)> &localBins,
               const std::tuple<Axes...> & /*axes*/, size_t &bin,
               size_t &area) {
    bin += area * localBins(0u);
  }

  template <class Point, class... Axes>
  ACTS_DEVICE_FUNC static void
  getLocalBinIndices(const Point &point, const std::tuple<Axes...> &axes,
                     ActsVector<size_t, sizeof...(Axes)> &indices) {
    const auto &thisAxis = std::get<0u>(axes);
    indices(0u) = thisAxis.getBin(point[0u]);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getLocalBinIndices(size_t &bin, const std::tuple<Axes...> & /*axes*/,
                     size_t &area,
                     ActsVector<size_t, sizeof...(Axes)> &indices) {
    // make sure to account for under-/overflow bins
    indices(0u) = bin / area;
    bin %= area;
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getLowerLeftBinEdge(ActsVector<ActsScalar, sizeof...(Axes)> &llEdge,
                      const ActsVector<size_t, sizeof...(Axes)> &localIndices,
                      const std::tuple<Axes...> &axes) {
    llEdge(0u) = std::get<0u>(axes).getBinLowerBound(localIndices(0u));
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getLowerLeftBinIndices(ActsVector<size_t, sizeof...(Axes)> &localIndices,
                         const std::tuple<Axes...> &axes) {
    localIndices(0u) = std::get<0u>(axes).wrapBin(localIndices(0u) - 1);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getNBins(const std::tuple<Axes...> &axes,
           ActsVector<size_t, sizeof...(Axes)> &nBinsArray) {
    // by convention getNBins does not include under-/overflow bins
    nBinsArray(0u) = std::get<0u>(axes).getNBins();
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getAxes(const std::tuple<Axes...> &axes,
          ActsVector<const IAxis *, sizeof...(Axes)> &axesArr) {
    axesArr(0u) = static_cast<const IAxis *>(&std::get<0u>(axes));
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getUpperRightBinEdge(ActsVector<ActsScalar, sizeof...(Axes)> &urEdge,
                       const ActsVector<size_t, sizeof...(Axes)> &localIndices,
                       const std::tuple<Axes...> &axes) {
    urEdge(0u) = std::get<0u>(axes).getBinUpperBound(localIndices(0u));
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getUpperRightBinIndices(ActsVector<size_t, sizeof...(Axes)> &localIndices,
                          const std::tuple<Axes...> &axes) {
    localIndices(0u) = std::get<0u>(axes).wrapBin(localIndices(0u) + 1);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getMin(const std::tuple<Axes...> &axes,
         ActsVector<ActsScalar, sizeof...(Axes)> &minArray) {
    minArray(0u) = std::get<0u>(axes).getMin();
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  getMax(const std::tuple<Axes...> &axes,
         ActsVector<ActsScalar, sizeof...(Axes)> &maxArray) {
    maxArray(0u) = std::get<0u>(axes).getMax();
  }

  template <class Point, class... Axes>
  ACTS_DEVICE_FUNC static bool isInside(const Point &position,
                                        const std::tuple<Axes...> &axes) {
    return std::get<0u>(axes).isInside(position[0u]);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void neighborHoodIndices(
      const ActsVector<size_t, sizeof...(Axes)> &localIndices,
      std::pair<size_t, size_t> sizes, const std::tuple<Axes...> &axes,
      ActsVector<NeighborHoodIndices, sizeof...(Axes)> &neighborIndices) {
    // ask 0-th axis
    size_t locIdx = localIndices(0u);
    NeighborHoodIndices locNeighbors =
        std::get<0u>(axes).neighborHoodIndices(locIdx, sizes);
    neighborIndices(0u) = locNeighbors;
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static void
  exteriorBinIndices(ActsVector<size_t, sizeof...(Axes)> &idx,
                     ActsVector<int, sizeof...(Axes)> isExterior,
                     std::set<size_t> &combinations,
                     const std::tuple<Axes...> &axes) {
    // For each exterior bin on this axis, we will do this
    auto recordExteriorBin = [&](size_t i) {
      idx(0u) = i;
      // at this point, combinations are complete: save the global bin
      size_t bin = 0, area = 1;
      grid_helper_impl<sizeof...(Axes) - 1>::getGlobalBin(idx, axes, bin, area);
      combinations.insert(bin);
    };

    // The first and last bins on this axis are exterior by definition
    for (size_t i :
         {static_cast<size_t>(0), std::get<0u>(axes).getNBins() + 1}) {
      recordExteriorBin(i);
    }

    // If no other axis is on an exterior index, stop here
    bool otherAxisExterior = false;
    for (size_t N = 1; N < sizeof...(Axes); ++N) {
      otherAxisExterior = otherAxisExterior | isExterior(N);
    }
    if (!otherAxisExterior) {
      return;
    }

    // Otherwise, we're on a grid border: iterate over all the other indices
    for (size_t i = 1; i <= std::get<0u>(axes).getNBins(); ++i) {
      recordExteriorBin(i);
    }
  }
};
/// @endcond

/// @brief helper functions for grid-related operations
struct grid_helper {
  /// @brief get the global indices for closest points on grid
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] bin  global bin index for bin of interest
  /// @param  [in] axes actual axis objects spanning the grid
  /// @return Sorted collection of global bin indices for bins whose
  ///         lower-left corners are the closest points on the grid to every
  ///         point in the given bin
  ///
  /// @note @c bin must be a valid bin index (excluding under-/overflow bins
  ///       along any axis).
  template <class... Axes>
  ACTS_DEVICE_FUNC static GlobalNeighborHoodIndices<sizeof...(Axes)>
  closestPointsIndices(const ActsVector<size_t, sizeof...(Axes)> &localIndices,
                       const std::tuple<Axes...> &axes) {
    // get neighboring bins, but only increment.
    return neighborHoodIndices(localIndices, std::make_pair(0, 1), axes);
  }

  /// @brief retrieve bin center from set of local bin indices
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] localIndices local bin indices along each axis
  /// @param  [in] axes         actual axis objects spanning the grid
  /// @return center position of bin
  ///
  /// @pre @c localIndices must only contain valid bin indices (i.e. excluding
  ///      under-/overflow bins).
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<ActsScalar, sizeof...(Axes)>
  getBinCenter(const ActsVector<size_t, sizeof...(Axes)> &localIndices,
               const std::tuple<Axes...> &axes) {
    ActsVector<ActsScalar, sizeof...(Axes)> center;
    constexpr size_t MAX = sizeof...(Axes) - 1;
    grid_helper_impl<MAX>::getBinCenter(center, localIndices, axes);

    return center;
  }

  /// @brief determine global bin index from local indices along each axis
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  ///
  /// @param  [in] localBins local bin indices along each axis
  /// @param  [in] axes  actual axis objects spanning the grid
  /// @return global index for bin defined by the local bin indices
  ///
  /// @pre All local bin indices must be a valid index for the corresponding
  ///      axis (including the under-/overflow bin for this axis).
  template <class... Axes>
  ACTS_DEVICE_FUNC static size_t
  getGlobalBin(const ActsVector<size_t, sizeof...(Axes)> &localBins,
               const std::tuple<Axes...> &axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;
    size_t area = 1;
    size_t bin = 0;

    grid_helper_impl<MAX>::getGlobalBin(localBins, axes, bin, area);

    return bin;
  }

  /// @brief determine local bin index for each axis from point
  ///
  /// @tparam Point any type with point semantics supporting component access
  ///               through @c operator[]
  /// @tparam Axes parameter pack of axis types defining the grid
  ///
  /// @param  [in] point point to look up in the grid
  /// @param  [in] axes  actual axis objects spanning the grid
  /// @return array with local bin indices along each axis (in same order as
  ///         given @c axes object)
  ///
  /// @pre The given @c Point type must represent a point in d (or higher)
  ///      dimensions where d is the number of axis objects in the tuple.
  /// @note This could be a under-/overflow bin along one or more axes.
  template <class Point, class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<size_t, sizeof...(Axes)>
  getLocalBinIndices(const Point &point, const std::tuple<Axes...> &axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;
    ActsVector<size_t, sizeof...(Axes)> indices;

    grid_helper_impl<MAX>::getLocalBinIndices(point, axes, indices);

    return indices;
  }

  /// @brief determine local bin index for each axis from global bin index
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  ///
  /// @param  [in] bin  global bin index
  /// @param  [in] axes actual axis objects spanning the grid
  /// @return array with local bin indices along each axis (in same order as
  ///         given @c axes object)
  ///
  /// @note Local bin indices can contain under-/overflow bins along any axis.
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<size_t, sizeof...(Axes)>
  getLocalBinIndices(size_t bin, const std::tuple<Axes...> &axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;
    size_t area = 1;
    ActsVector<size_t, sizeof...(Axes)> indices;

    grid_helper_impl<MAX>::getLocalBinIndices(bin, axes, area, indices);

    return indices;
  }

  /// @brief retrieve lower-left bin edge from set of local bin indices
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] localIndices local bin indices along each axis
  /// @param  [in] axes         actual axis objects spanning the grid
  /// @return generalized lower-left bin edge
  ///
  /// @pre @c localIndices must only contain valid bin indices (excluding
  ///      underflow bins).
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<ActsScalar, sizeof...(Axes)>
  getLowerLeftBinEdge(const ActsVector<size_t, sizeof...(Axes)> &localIndices,
                      const std::tuple<Axes...> &axes) {
    ActsVector<ActsScalar, sizeof...(Axes)> llEdge;
    constexpr size_t MAX = sizeof...(Axes) - 1;
    grid_helper_impl<MAX>::getLowerLeftBinEdge(llEdge, localIndices, axes);

    return llEdge;
  }

  /// @brief get local bin indices for lower-left neighboring bin
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] localIndices local bin indices along each axis
  /// @param  [in] axes         actual axis objects spanning the grid
  /// @return array with local bin indices of lower-left neighbor bin
  ///
  /// @pre @c localIndices must only contain valid bin indices (excluding
  ///      underflow bins).
  ///
  /// This function returns the local bin indices for the generalized
  /// lower-left neighbor which simply means that all local bin indices are
  /// decremented by one.
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<size_t, sizeof...(Axes)>
  getLowerLeftBinIndices(
      const ActsVector<size_t, sizeof...(Axes)> &localIndices,
      const std::tuple<Axes...> &axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;
    auto llIndices = localIndices;
    grid_helper_impl<MAX>::getLowerLeftBinIndices(llIndices, axes);

    return llIndices;
  }

  /// @brief calculate number of bins in a grid defined by a set of
  /// axes for each axis
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] axes actual axis objects spanning the grid
  /// @return array of number of bins for each axis of the grid
  ///
  /// @note This does not include under-/overflow bins along each axis.
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<size_t, sizeof...(Axes)>
  getNBins(const std::tuple<Axes...> &axes) {
    ActsVector<size_t, sizeof...(Axes)> nBinsArray;
    grid_helper_impl<sizeof...(Axes) - 1>::getNBins(axes, nBinsArray);
    return nBinsArray;
  }

  /// @brief return an array with copies of the axes, converted
  /// to type AnyAxis
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param [in] axes actual axis objects spanning the grid
  /// @return array with copies of the axis
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<const IAxis *, sizeof...(Axes)>
  getAxes(const std::tuple<Axes...> &axes) {
    ActsVector<const IAxis *, sizeof...(Axes)> arr;
    grid_helper_impl<sizeof...(Axes) - 1>::getAxes(axes, arr);
    return arr;
  }

  /// @brief retrieve upper-right bin edge from set of local bin indices
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] localIndices local bin indices along each axis
  /// @param  [in] axes         actual axis objects spanning the grid
  /// @return generalized upper-right bin edge
  ///
  /// @pre @c localIndices must only contain valid bin indices (excluding
  ///      overflow bins).
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<ActsScalar, sizeof...(Axes)>
  getUpperRightBinEdge(const ActsVector<size_t, sizeof...(Axes)> &localIndices,
                       const std::tuple<Axes...> &axes) {
    ActsVector<ActsScalar, sizeof...(Axes)> urEdge;
    constexpr size_t MAX = sizeof...(Axes) - 1;
    grid_helper_impl<MAX>::getUpperRightBinEdge(urEdge, localIndices, axes);

    return urEdge;
  }

  /// @brief get local bin indices for upper-right neighboring bin
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] localIndices local bin indices along each axis
  /// @param  [in] axes         actual axis objects spanning the grid
  /// @return array with local bin indices of upper-right neighbor bin
  ///
  /// @pre @c localIndices must only contain valid bin indices (excluding
  ///      overflow bins).
  ///
  /// This function returns the local bin indices for the generalized
  /// upper-right neighbor which simply means that all local bin indices are
  /// incremented by one.
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<size_t, sizeof...(Axes)>
  getUpperRightBinIndices(
      const ActsVector<size_t, sizeof...(Axes)> &localIndices,
      const std::tuple<Axes...> &axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;
    auto urIndices = localIndices;
    grid_helper_impl<MAX>::getUpperRightBinIndices(urIndices, axes);

    return urIndices;
  }

  /// @brief get the minimum value of all axes of one grid
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] axes actual axis objects spanning the grid
  /// @return array returning the minima of all given axes
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<ActsScalar, sizeof...(Axes)>
  getMin(const std::tuple<Axes...> &axes) {
    ActsVector<ActsScalar, sizeof...(Axes)> minArray;
    grid_helper_impl<sizeof...(Axes) - 1>::getMin(axes, minArray);
    return minArray;
  }

  /// @brief get the maximum value of all axes of one grid
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] axes actual axis objects spanning the grid
  /// @return array returning the maxima of all given axes
  template <class... Axes>
  ACTS_DEVICE_FUNC static ActsVector<ActsScalar, sizeof...(Axes)>
  getMax(const std::tuple<Axes...> &axes) {
    ActsVector<ActsScalar, sizeof...(Axes)> maxArray;
    grid_helper_impl<sizeof...(Axes) - 1>::getMax(axes, maxArray);
    return maxArray;
  }

  /// @brief get global bin indices for bins in specified neighborhood
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] localIndices local bin indices along each axis
  /// @param  [in] size         size of neighborhood determining how many
  ///                           adjacent bins along each axis are considered
  /// @param  [in] axes         actual axis objects spanning the grid
  /// @return Sorted collection of global bin indices for all bins in
  ///         the neighborhood
  ///
  /// @note Over-/underflow bins are included in the neighborhood.
  /// @note The @c size parameter sets the range by how many units each local
  ///       bin index is allowed to be varied. All local bin indices are
  ///       varied independently, that is diagonal neighbors are included.
  ///       Ignoring the truncation of the neighborhood size reaching beyond
  ///       over-/underflow bins, the neighborhood is of size \f$2 \times
  ///       \text{size}+1\f$ along each dimension.
  /// @note The concrete bins which are returned depend on the WrappingTypes
  ///       of the contained axes
  ///
  template <class... Axes>
  ACTS_DEVICE_FUNC static GlobalNeighborHoodIndices<sizeof...(Axes)>
  neighborHoodIndices(const ActsVector<size_t, sizeof...(Axes)> &localIndices,
                      std::pair<size_t, size_t> sizes,
                      const std::tuple<Axes...> &axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;

    // length N array which contains local neighbors based on size par
    ActsVector<NeighborHoodIndices, sizeof...(Axes)> neighborIndices;
    // get local bin indices for neighboring bins
    grid_helper_impl<MAX>::neighborHoodIndices(localIndices, sizes, axes,
                                               neighborIndices);

    // Query the number of bins
    ActsVector<size_t, sizeof...(Axes)> nBinsArray = getNBins(axes);

    // Produce iterator of global indices
    return GlobalNeighborHoodIndices<sizeof...(Axes)>(neighborIndices,
                                                      nBinsArray);
  }

  template <class... Axes>
  ACTS_DEVICE_FUNC static GlobalNeighborHoodIndices<sizeof...(Axes)>
  neighborHoodIndices(const ActsVector<size_t, sizeof...(Axes)> &localIndices,
                      size_t size, const std::tuple<Axes...> &axes) {
    return neighborHoodIndices(localIndices, std::make_pair(size, size), axes);
  }

  /// @brief get bin indices of all overflow and underflow bins
  ///
  /// @tparam Axes parameter pack of axis types defining the grid
  /// @param  [in] axes         actual axis objects spanning the grid
  /// @return set of global bin indices for all over- and underflow bins
  template <class... Axes>
  ACTS_DEVICE_FUNC static std::set<size_t>
  exteriorBinIndices(const std::tuple<Axes...> &axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;

    ActsVector<size_t, sizeof...(Axes)> idx;
    ActsVector<int, sizeof...(Axes)> isExterior;
    std::set<size_t> combinations;
    grid_helper_impl<MAX>::exteriorBinIndices(idx, isExterior, combinations,
                                              axes);

    return combinations;
  }

  /// @brief check whether given point is inside axes limits
  ///
  /// @tparam Point any type with point semantics supporting component access
  ///               through @c operator[]
  /// @tparam Axes parameter pack of axis types defining the grid
  ///
  /// @param  [in] position point to look up in the grid
  /// @param  [in] axes     actual axis objects spanning the grid
  /// @return @c true if \f$\text{xmin_i} \le x_i < \text{xmax}_i \forall i=0,
  ///         \dots, d-1\f$, otherwise @c false
  ///
  /// @pre The given @c Point type must represent a point in d (or higher)
  ///      dimensions where d is the number of axis objects in the tuple.
  template <class Point, class... Axes>
  ACTS_DEVICE_FUNC static bool isInside(const Point &position,
                                        const std::tuple<Axes...> &axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;
    return grid_helper_impl<MAX>::isInside(position, axes);
  }
};

} // namespace detail

} // namespace Acts
