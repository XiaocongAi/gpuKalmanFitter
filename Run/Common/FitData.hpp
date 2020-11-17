#pragma once

#include "EventData/PixelSourceLink.hpp"
#include "EventData/TrackParameters.hpp"
#include "Fitter/GainMatrixSmoother.hpp"
#include "Fitter/GainMatrixUpdater.hpp"
#include "Fitter/KalmanFitter.hpp"
#include "Propagator/EigenStepper.hpp"
#include "Propagator/Propagator.hpp"
#include "Surfaces/LineSurface.hpp"
#include "Surfaces/PlaneSurface.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Units.hpp"

#include "ActsExamples/Generator.hpp"
#include "ActsExamples/RandomNumbers.hpp"

#include "ActsFatras/EventData/Barcode.hpp"
#include "ActsFatras/EventData/Particle.hpp"
#include "ActsFatras/Kernel/MinimalSimulator.hpp"

#include "Test/Helper.hpp"

#pragma once

using Simulator = ActsFatras::MinimalSimulator<ActsExamples::RandomEngine>;
using PlaneSurfaceType = Acts::PlaneSurface<Acts::InfiniteBounds>;
using Stepper = Acts::EigenStepper<Test::ConstantBField>;
using PropagatorType = Acts::Propagator<Stepper>;
using PropResultType = Acts::PropagatorResult;
using PropOptionsType = Acts::PropagatorOptions<Simulator, Test::VoidAborter>;
using Smoother = GainMatrixSmoother<Acts::BoundParameters<PlaneSurfaceType>>;
using KalmanFitterType =
    Acts::KalmanFitter<PropagatorType, Acts::GainMatrixUpdater, Smoother>;
using KalmanFitterResultType =
    Acts::KalmanFitterResult<Acts::PixelSourceLink,
                             Acts::BoundParameters<PlaneSurfaceType>,
                             Acts::LineSurface>;
using TSType = typename KalmanFitterResultType::TrackStateType;
using FitOptionsType = Acts::KalmanFitterOptions<Acts::VoidOutlierFinder>;

using Size = unsigned int;

struct BoundState {
  Acts::BoundVector boundParams;
  Acts::BoundMatrix boundCov;
};
