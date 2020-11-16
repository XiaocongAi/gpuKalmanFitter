// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EventData/ChargePolicy.hpp"
#include "EventData/SingleBoundTrackParameters.hpp"
#include "EventData/SingleCurvilinearTrackParameters.hpp"

namespace Acts {
template <typename surface_derived_t = PlaneSurface<InfiniteBounds>>
using BoundParameters =
    SingleBoundTrackParameters<ChargedPolicy, surface_derived_t>;

using CurvilinearParameters = SingleCurvilinearTrackParameters<ChargedPolicy>;

} // namespace Acts
