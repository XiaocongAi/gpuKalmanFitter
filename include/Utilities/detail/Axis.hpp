// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Utilities/Definitions.hpp"
#include "Utilities/IAxis.hpp"
#include "Utilities/detail/AxisFwd.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace Acts {

namespace detail {

// This object can be iterated to produce up to two sequences of integer
// indices, corresponding to the half-open integer ranges [begin1, end1[ and
// [begin2, end2[.
//
// The goal is to emulate the effect of enumerating a range of neighbor
// indices on an axis (which may go out of bounds and wrap around since we
// have AxisBoundaryType::Closed), inserting them into an ActsXVector, and
// discarding duplicates, without paying the price of duplicate removal
// and dynamic memory allocation in hot magnetic field interpolation code.
//
class NeighborHoodIndices {
public:
  NeighborHoodIndices() = default;

  ACTS_DEVICE_FUNC NeighborHoodIndices(size_t begin, size_t end)
      : m_begin1(begin), m_end1(end), m_begin2(end), m_end2(end) {}

  ACTS_DEVICE_FUNC NeighborHoodIndices(size_t begin1, size_t end1,
                                       size_t begin2, size_t end2)
      : m_begin1(begin1), m_end1(end1), m_begin2(begin2), m_end2(end2) {}

  class iterator {
  public:
    iterator() = default;

    // Specialized constructor for end() iterator
    ACTS_DEVICE_FUNC iterator(size_t current)
        : m_current(current), m_wrapped(true) {}

    ACTS_DEVICE_FUNC iterator(size_t begin1, size_t end1, size_t begin2)
        : m_current(begin1), m_end1(end1), m_begin2(begin2),
          m_wrapped(begin1 == begin2) {}

    ACTS_DEVICE_FUNC size_t operator*() const { return m_current; }

    ACTS_DEVICE_FUNC iterator &operator++() {
      ++m_current;
      if (m_current == m_end1) {
        m_current = m_begin2;
        m_wrapped = true;
      }
      return *this;
    }

    ACTS_DEVICE_FUNC bool operator==(const iterator &it) const {
      return (m_current == it.m_current) && (m_wrapped == it.m_wrapped);
    }

    ACTS_DEVICE_FUNC bool operator!=(const iterator &it) const {
      return !(*this == it);
    }

  private:
    size_t m_current, m_end1, m_begin2;
    bool m_wrapped;
  };

  ACTS_DEVICE_FUNC iterator begin() const {
    return iterator(m_begin1, m_end1, m_begin2);
  }

  ACTS_DEVICE_FUNC iterator end() const { return iterator(m_end2); }

  // Number of indices that will be produced if this sequence is iterated
  ACTS_DEVICE_FUNC size_t size() const {
    return (m_end1 - m_begin1) + (m_end2 - m_begin2);
  }

  // Collect the sequence of indices into an ActsXVector
  ACTS_DEVICE_FUNC ActsXVector<size_t> collect() const {
    ActsXVector<size_t> result;
    result.resize(this->size());
    size_t iRow = 0;
    for (size_t idx : *this) {
      result(iRow) = idx;
      iRow++;
    }
    return result;
  }

private:
  size_t m_begin1 = 0, m_end1 = 0, m_begin2 = 0, m_end2 = 0;
};

/// @brief calculate bin indices for an equidistant binning
///
/// This class provides some basic functionality for calculating bin indices
/// for a given equidistant binning.
template <AxisBoundaryType bdt>
class Axis<AxisType::Equidistant, bdt> final : public IAxis {
public:
  /// @brief default constructor
  ///
  /// @param [in] xmin lower boundary of axis range
  /// @param [in] xmax upper boundary of axis range
  /// @param [in] nBins number of bins to divide the axis range into
  ///
  /// Divide the range \f$[\text{xmin},\text{xmax})\f$ into \f$\text{nBins}\f$
  /// equidistant bins.
  ACTS_DEVICE_FUNC Axis(double xmin, double xmax, size_t nBins)
      : m_min(xmin), m_max(xmax), m_width((xmax - xmin) / nBins),
        m_bins(nBins) {}

  /// @brief returns whether the axis is equidistant
  ///
  /// @return bool is equidistant
  ACTS_DEVICE_FUNC bool isEquidistant() const override { return true; }

  /// @brief returns whether the axis is variable
  ///
  /// @return bool is variable
  ACTS_DEVICE_FUNC bool isVariable() const override { return false; }

  /// @brief returns the boundary type set in the template param
  ///
  /// @return @c AxisBoundaryType of this axis
  ACTS_DEVICE_FUNC AxisBoundaryType getBoundaryType() const override {
    return bdt;
  }

  /// @brief Get #size bins which neighbor the one given
  ///
  /// Generic overload with symmetric size
  ///
  /// @param [in] idx requested bin index
  /// @param [in] sizes how many neighboring bins (up/down)
  /// @return Set of neighboring bin indices (global)
  ACTS_DEVICE_FUNC NeighborHoodIndices
  neighborHoodIndices(size_t idx, size_t size = 1) const {
    return neighborHoodIndices(idx, std::make_pair(size, size));
  }

  /// @brief Get #size bins which neighbor the one given
  ///
  /// This is the version for Open
  ///
  /// @param [in] idx requested bin index
  /// @param [in] sizes how many neighboring bins (up/down)
  /// @return Set of neighboring bin indices (global)
  /// @note Open varies given bin and allows 0 and NBins+1 (underflow,
  /// overflow)
  ///       as neighbors
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Open, int> = 0>
  ACTS_DEVICE_FUNC NeighborHoodIndices neighborHoodIndices(
      size_t idx, std::pair<size_t, size_t> sizes = {1, 1}) const {
    constexpr int min = 0;
    const int max = getNBins() + 1;
    const int itmin = std::max(min, int(idx - sizes.first));
    const int itmax = std::min(max, int(idx + sizes.second));
    return NeighborHoodIndices(itmin, itmax + 1);
  }

  /// @brief Get #size bins which neighbor the one given
  ///
  /// This is the version for Bound
  ///
  /// @param [in] idx requested bin index
  /// @param [in] sizes how many neighboring bins (up/down)
  /// @return Set of neighboring bin indices (global)
  /// @note Bound varies given bin and allows 1 and NBins (regular bins)
  ///       as neighbors
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Bound, int> = 0>
  ACTS_DEVICE_FUNC NeighborHoodIndices neighborHoodIndices(
      size_t idx, std::pair<size_t, size_t> sizes = {1, 1}) const {
    if (idx <= 0 || idx >= (getNBins() + 1)) {
      return NeighborHoodIndices();
    }
    constexpr int min = 1;
    const int max = getNBins();
    const int itmin = std::max(min, int(idx - sizes.first));
    const int itmax = std::min(max, int(idx + sizes.second));
    return NeighborHoodIndices(itmin, itmax + 1);
  }

  /// @brief Get #size bins which neighbor the one given
  ///
  /// This is the version for Closed
  ///
  /// @param [in] idx requested bin index
  /// @param [in] sizes how many neighboring bins (up/down)
  /// @return Set of neighboring bin indices (global)
  /// @note Closed varies given bin and allows bins on the opposite
  ///       side of the axis as neighbors. (excludes underflow / overflow)
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Closed, int> = 0>
  ACTS_DEVICE_FUNC NeighborHoodIndices neighborHoodIndices(
      size_t idx, std::pair<size_t, size_t> sizes = {1, 1}) const {
    // Handle invalid indices
    if (idx <= 0 || idx >= (getNBins() + 1)) {
      return NeighborHoodIndices();
    }

    // Handle corner case where user requests more neighbours than the number
    // of bins on the axis. We do not want to double-count bins in that case.
    sizes.first %= getNBins();
    sizes.second %= getNBins();
    if (sizes.first + sizes.second + 1 > getNBins()) {
      sizes.second -= (sizes.first + sizes.second + 1) - getNBins();
    }

    // If the entire index range is not covered, we must wrap the range of
    // targeted neighbor indices into the range of valid bin indices. This may
    // split the range of neighbor indices in two parts:
    //
    // Before wraparound - [        XXXXX]XXX
    // After wraparound  - [ XXXX   XXXX ]
    //
    const int itmin = idx - sizes.first;
    const int itmax = idx + sizes.second;
    const size_t itfirst = wrapBin(itmin);
    const size_t itlast = wrapBin(itmax);
    if (itfirst <= itlast) {
      return NeighborHoodIndices(itfirst, itlast + 1);
    } else {
      return NeighborHoodIndices(itfirst, getNBins() + 1, 1, itlast + 1);
    }
  }

  /// @brief Converts bin index into a valid one for this axis.
  ///
  /// @note Open: bin index is clamped to [0, nBins+1]
  ///
  /// @param [in] bin The bin to wrap
  /// @return valid bin index
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Open, int> = 0>
  ACTS_DEVICE_FUNC size_t wrapBin(int bin) const {
    return std::max(std::min(bin, static_cast<int>(getNBins()) + 1), 0);
  }

  /// @brief Converts bin index into a valid one for this axis.
  ///
  /// @note Bound: bin index is clamped to [1, nBins]
  ///
  /// @param [in] bin The bin to wrap
  /// @return valid bin index
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Bound, int> = 0>
  ACTS_DEVICE_FUNC size_t wrapBin(int bin) const {
    return std::max(std::min(bin, static_cast<int>(getNBins())), 1);
  }

  /// @brief Converts bin index into a valid one for this axis.
  ///
  /// @note Closed: bin index wraps around to other side
  ///
  /// @param [in] bin The bin to wrap
  /// @return valid bin index
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Closed, int> = 0>
  ACTS_DEVICE_FUNC size_t wrapBin(int bin) const {
    const int w = getNBins();
    return 1 + (w + ((bin - 1) % w)) % w;
    // return int(bin<1)*w - int(bin>w)*w + bin;
  }

  /// @brief get corresponding bin index for given coordinate
  ///
  /// @param  [in] x input coordinate
  /// @return index of bin containing the given value
  ///
  /// @note Bin intervals are defined with closed lower bounds and open upper
  ///       bounds, that is \f$l <= x < u\f$ if the value @c x lies within a
  ///       bin with lower bound @c l and upper bound @c u.
  /// @note Bin indices start at @c 1. The underflow bin has the index @c 0
  ///       while the index <tt>nBins + 1</tt> indicates the overflow bin .
  ACTS_DEVICE_FUNC size_t getBin(double x) const {
    return wrapBin(std::floor((x - getMin()) / getBinWidth()) + 1);
  }

  /// @brief get bin width
  ///
  /// @return constant width for all bins
  ACTS_DEVICE_FUNC double getBinWidth(size_t /*bin*/ = 0) const {
    return m_width;
  }

  /// @brief get lower bound of bin
  ///
  /// @param  [in] bin index of bin
  /// @return lower bin boundary
  ///
  /// @pre @c bin must be a valid bin index (excluding the underflow bin),
  ///      i.e. \f$1 \le \text{bin} \le \text{nBins} + 1\f$
  ///
  /// @note Bin intervals have a closed lower bound, i.e. the lower boundary
  ///       belongs to the bin with the given bin index.
  ACTS_DEVICE_FUNC double getBinLowerBound(size_t bin) const {
    return getMin() + (bin - 1) * getBinWidth();
  }

  /// @brief get upper bound of bin
  ///
  /// @param  [in] bin index of bin
  /// @return upper bin boundary
  ///
  /// @pre @c bin must be a valid bin index (excluding the overflow bin),
  ///      i.e. \f$0 \le \text{bin} \le \text{nBins}\f$
  ///
  /// @note Bin intervals have an open upper bound, i.e. the upper boundary
  ///       does @b not belong to the bin with the given bin index.
  ACTS_DEVICE_FUNC double getBinUpperBound(size_t bin) const {
    return getMin() + bin * getBinWidth();
  }

  /// @brief get bin center
  ///
  /// @param  [in] bin index of bin
  /// @return bin center position
  ///
  /// @pre @c bin must be a valid bin index (excluding under-/overflow bins),
  ///      i.e. \f$1 \le \text{bin} \le \text{nBins}\f$
  ACTS_DEVICE_FUNC double getBinCenter(size_t bin) const {
    return getMin() + (bin - 0.5) * getBinWidth();
  }

  /// @brief get maximum of binning range
  ///
  /// @return maximum of binning range
  ACTS_DEVICE_FUNC double getMax() const override { return m_max; }

  /// @brief get minimum of binning range
  ///
  /// @return minimum of binning range
  ACTS_DEVICE_FUNC double getMin() const override { return m_min; }

  /// @brief get total number of bins
  ///
  /// @return total number of bins (excluding under-/overflow bins)
  ACTS_DEVICE_FUNC size_t getNBins() const override { return m_bins; }

  /// @brief check whether value is inside axis limits
  ///
  /// @return @c true if \f$\text{xmin} \le x < \text{xmax}\f$, otherwise
  ///         @c false
  ///
  /// @post If @c true is returned, the bin containing the given value is a
  ///       valid bin, i.e. it is neither the underflow nor the overflow bin.
  ACTS_DEVICE_FUNC bool isInside(double x) const {
    return (m_min <= x) && (x < m_max);
  }

  /// @brief Return a vector of bin edges
  /// @return Vector which contains the bin edges
  ACTS_DEVICE_FUNC ActsXVector<double> getBinEdges() const override {
    ActsXVector<double> binEdges;
    binEdges.resize(m_bins + 1);
    for (size_t i = 1; i <= m_bins; i++) {
      binEdges(i - 1) = getBinLowerBound(i);
    }
    binEdges(m_bins) = getBinUpperBound(m_bins);
    return binEdges;
  }

private:
  /// minimum of binning range
  double m_min;
  /// maximum of binning range
  double m_max;
  /// constant bin width
  double m_width;
  /// number of bins (excluding under-/overflow bins)
  size_t m_bins;
};

/// @brief calculate bin indices for a variable binning
///
/// This class provides some basic functionality for calculating bin indices
/// for a given binning with variable bin sizes.
template <AxisBoundaryType bdt>
class Axis<AxisType::Variable, bdt> final : public IAxis {
public:
  /// @brief default constructor
  ///
  /// @param [in] binEdges vector of bin edges
  /// @pre @c binEdges must be strictly sorted in ascending order.
  /// @pre @c binEdges must contain at least two entries.
  ///
  /// Create a binning structure with @c nBins variable-sized bins from the
  /// given bin boundaries. @c nBins is given by the number of bin edges
  /// reduced by one.
  ACTS_DEVICE_FUNC Axis(ActsXVector<double> binEdges)
      : m_binEdges(std::move(binEdges)) {}

  /// @brief returns whether the axis is equidistante
  ///
  /// @return bool is equidistant
  ACTS_DEVICE_FUNC bool isEquidistant() const override { return false; }

  /// @brief returns whether the axis is variable
  ///
  /// @return bool is variable
  ACTS_DEVICE_FUNC bool isVariable() const override { return true; }

  /// @brief returns the boundary type set in the template param
  ///
  /// @return @c AxisBoundaryType of this axis
  ACTS_DEVICE_FUNC AxisBoundaryType getBoundaryType() const override {
    return bdt;
  }

  /// @brief Get #size bins which neighbor the one given
  ///
  /// Generic overload with symmetric size
  ///
  /// @param [in] idx requested bin index
  /// @param [in] size how many neighboring bins
  /// @return Set of neighboring bin indices (global)
  ACTS_DEVICE_FUNC NeighborHoodIndices
  neighborHoodIndices(size_t idx, size_t size = 1) const {
    return neighborHoodIndices(idx, std::make_pair(size, size));
  }

  /// @brief Get #size bins which neighbor the one given
  ///
  /// This is the version for Open
  ///
  /// @param [in] idx requested bin index
  /// @param [in] sizes how many neighboring bins (up/down)
  /// @return Set of neighboring bin indices (global)
  /// @note Open varies given bin and allows 0 and NBins+1 (underflow,
  /// overflow)
  ///       as neighbors
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Open, int> = 0>
  ACTS_DEVICE_FUNC NeighborHoodIndices neighborHoodIndices(
      size_t idx, std::pair<size_t, size_t> sizes = {1, 1}) const {
    constexpr int min = 0;
    const int max = getNBins() + 1;
    const int itmin = std::max(min, int(idx - sizes.first));
    const int itmax = std::min(max, int(idx + sizes.second));
    return NeighborHoodIndices(itmin, itmax + 1);
  }

  /// @brief Get #size bins which neighbor the one given
  ///
  /// This is the version for Bound
  ///
  /// @param [in] idx requested bin index
  /// @param [in] sizes how many neighboring bins (up/down)
  /// @return Set of neighboring bin indices (global)
  /// @note Bound varies given bin and allows 1 and NBins (regular bins)
  ///       as neighbors
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Bound, int> = 0>
  ACTS_DEVICE_FUNC NeighborHoodIndices neighborHoodIndices(
      size_t idx, std::pair<size_t, size_t> sizes = {1, 1}) const {
    if (idx <= 0 || idx >= (getNBins() + 1)) {
      return NeighborHoodIndices();
    }
    constexpr int min = 1;
    const int max = getNBins();
    const int itmin = std::max(min, int(idx - sizes.first));
    const int itmax = std::min(max, int(idx + sizes.second));
    return NeighborHoodIndices(itmin, itmax + 1);
  }

  /// @brief Get #size bins which neighbor the one given
  ///
  /// This is the version for Closed
  ///
  /// @param [in] idx requested bin index
  /// @param [in] sizes how many neighboring bins (up/down)
  /// @return Set of neighboring bin indices (global)
  /// @note Closed varies given bin and allows bins on the opposite
  ///       side of the axis as neighbors. (excludes underflow / overflow)
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Closed, int> = 0>
  ACTS_DEVICE_FUNC NeighborHoodIndices neighborHoodIndices(
      size_t idx, std::pair<size_t, size_t> sizes = {1, 1}) const {
    // Handle invalid indices
    if (idx <= 0 || idx >= (getNBins() + 1)) {
      return NeighborHoodIndices();
    }

    // Handle corner case where user requests more neighbours than the number
    // of bins on the axis. We do not want to double-count bins in that case.
    sizes.first %= getNBins();
    sizes.second %= getNBins();
    if (sizes.first + sizes.second + 1 > getNBins()) {
      sizes.second -= (sizes.first + sizes.second + 1) - getNBins();
    }

    // If the entire index range is not covered, we must wrap the range of
    // targeted neighbor indices into the range of valid bin indices. This may
    // split the range of neighbor indices in two parts:
    //
    // Before wraparound - [        XXXXX]XXX
    // After wraparound  - [ XXXX   XXXX ]
    //
    const int itmin = idx - sizes.first;
    const int itmax = idx + sizes.second;
    const size_t itfirst = wrapBin(itmin);
    const size_t itlast = wrapBin(itmax);
    if (itfirst <= itlast) {
      return NeighborHoodIndices(itfirst, itlast + 1);
    } else {
      return NeighborHoodIndices(itfirst, getNBins() + 1, 1, itlast + 1);
    }
  }

  /// @brief Converts bin index into a valid one for this axis.
  ///
  /// @note Open: bin index is clamped to [0, nBins+1]
  ///
  /// @param [in] bin The bin to wrap
  /// @return valid bin index
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Open, int> = 0>
  ACTS_DEVICE_FUNC size_t wrapBin(int bin) const {
    return std::max(std::min(bin, static_cast<int>(getNBins()) + 1), 0);
  }

  /// @brief Converts bin index into a valid one for this axis.
  ///
  /// @note Bound: bin index is clamped to [1, nBins]
  ///
  /// @param [in] bin The bin to wrap
  /// @return valid bin index
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Bound, int> = 0>
  ACTS_DEVICE_FUNC size_t wrapBin(int bin) const {
    return std::max(std::min(bin, static_cast<int>(getNBins())), 1);
  }

  /// @brief Converts bin index into a valid one for this axis.
  ///
  /// @note Closed: bin index wraps around to other side
  ///
  /// @param [in] bin The bin to wrap
  /// @return valid bin index
  template <AxisBoundaryType T = bdt,
            std::enable_if_t<T == AxisBoundaryType::Closed, int> = 0>
  ACTS_DEVICE_FUNC size_t wrapBin(int bin) const {
    const int w = getNBins();
    return 1 + (w + ((bin - 1) % w)) % w;
    // return int(bin<1)*w - int(bin>w)*w + bin;
  }

  /// @brief get corresponding bin index for given coordinate
  ///
  /// @param  [in] x input coordinate
  /// @return index of bin containing the given value
  ///
  /// @note Bin intervals are defined with closed lower bounds and open upper
  ///       bounds, that is \f$l <= x < u\f$ if the value @c x lies within a
  ///       bin with lower bound @c l and upper bound @c u.
  /// @note Bin indices start at @c 1. The underflow bin has the index @c 0
  ///       while the index <tt>nBins + 1</tt> indicates the overflow bin .
  ACTS_DEVICE_FUNC size_t getBin(double x) const {

    size_t idx = 0;
    bool foundRow = false;
    for (; idx < m_binEdges.size(); idx++) {
      if (m_binEdges(idx) == x) {
        foundRow = true;
      }
      if (m_binEdges(idx) > x and foundRow) {
        break;
      }
    }
    return wrapBin(idx);
  }

  /// @brief get bin width
  ///
  /// @param  [in] bin index of bin
  /// @return width of given bin
  ///
  /// @pre @c bin must be a valid bin index (excluding under-/overflow bins),
  ///      i.e. \f$1 \le \text{bin} \le \text{nBins}\f$
  ACTS_DEVICE_FUNC double getBinWidth(size_t bin) const {
    return m_binEdges(bin) - m_binEdges(bin - 1);
  }

  /// @brief get lower bound of bin
  ///
  /// @param  [in] bin index of bin
  /// @return lower bin boundary
  ///
  /// @pre @c bin must be a valid bin index (excluding the underflow bin),
  ///      i.e. \f$1 \le \text{bin} \le \text{nBins} + 1\f$
  ///
  /// @note Bin intervals have a closed lower bound, i.e. the lower boundary
  ///       belongs to the bin with the given bin index.
  ACTS_DEVICE_FUNC double getBinLowerBound(size_t bin) const {
    return m_binEdges(bin - 1);
  }

  /// @brief get upper bound of bin
  ///
  /// @param  [in] bin index of bin
  /// @return upper bin boundary
  ///
  /// @pre @c bin must be a valid bin index (excluding the overflow bin),
  ///      i.e. \f$0 \le \text{bin} \le \text{nBins}\f$
  ///
  /// @note Bin intervals have an open upper bound, i.e. the upper boundary
  ///       does @b not belong to the bin with the given bin index.
  ACTS_DEVICE_FUNC double getBinUpperBound(size_t bin) const {
    return m_binEdges(bin);
  }

  /// @brief get bin center
  ///
  /// @param  [in] bin index of bin
  /// @return bin center position
  ///
  /// @pre @c bin must be a valid bin index (excluding under-/overflow bins),
  ///      i.e. \f$1 \le \text{bin} \le \text{nBins}\f$
  ACTS_DEVICE_FUNC double getBinCenter(size_t bin) const {
    return 0.5 * (getBinLowerBound(bin) + getBinUpperBound(bin));
  }

  /// @brief get maximum of binning range
  ///
  /// @return maximum of binning range
  ACTS_DEVICE_FUNC double getMax() const override {
    return m_binEdges(m_binEdges.size() - 1);
  }

  /// @brief get minimum of binning range
  ///
  /// @return minimum of binning range
  ACTS_DEVICE_FUNC double getMin() const override { return m_binEdges(0u); }

  /// @brief get total number of bins
  ///
  /// @return total number of bins (excluding under-/overflow bins)
  ACTS_DEVICE_FUNC size_t getNBins() const override {
    return m_binEdges.size() - 1;
  }

  /// @brief check whether value is inside axis limits
  ///
  /// @return @c true if \f$\text{xmin} \le x < \text{xmax}\f$, otherwise
  ///         @c false
  ///
  /// @post If @c true is returned, the bin containing the given value is a
  ///       valid bin, i.e. it is neither the underflow nor the overflow bin.
  ACTS_DEVICE_FUNC bool isInside(double x) const {
    double lowerEdge = m_binEdges(0u);
    double upperEdge = m_binEdges(m_binEdges.size() - 1);
    return (lowerEdge <= x) && (x < upperEdge);
  }

  /// @brief Return a vector of bin edges
  /// @return Vector which contains the bin edges
  ACTS_DEVICE_FUNC ActsXVector<double> getBinEdges() const override {
    return m_binEdges;
  }

private:
  /// vector of bin edges (sorted in ascending order)
  ActsXVector<double> m_binEdges;
};
} // namespace detail

} // namespace Acts
