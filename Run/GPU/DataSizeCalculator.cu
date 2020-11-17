#pragma once

#include "FitData.hpp"

enum FitData : int {
  SourceLinks = 0,
  StartState = 1,
  TargetSurface = 2,
  FitOptions = 3,
  FitStates = 4,
  FitParams = 5,
  FitStatus = 6
};

namespace FitDataSizeCalculator {
//  FitDataHelper(Size nTracks):numTracks(nTracks){}

std::array<Size, 7> totalBytes(Size nSurfaces, Size nTracks) {
  std::array<Size, 7> fitData;
  fitData[0] = sizeof(Acts::PixelSourceLink) * nSurfaces * nTracks;
  fitData[1] = sizeof(BoundState) * nTracks;
  fitData[2] = sizeof(Acts::LineSurface) * nTracks;
  fitData[3] = sizeof(FitOptionsType) * nTracks;
  fitData[4] = sizeof(TSType) * nSurfaces * nTracks;
  fitData[5] = sizeof(Acts::BoundParameters<Acts::LineSurface>) * nTracks;
  fitData[6] = sizeof(bool) * nTracks;
  return fitData;
}

std::array<Size, 8> streamBytes(Size nSurfaces, Size nTracks, Size nStreams,
                                Size iStream) {
  if (iStream >= nStreams) {
    throw std::invalid_argument("There are only " + nStreams);
  }

  Size tracksPerStream = (nTracks + nStreams - 1) / nStreams;
  Size tracksLastStream =
      tracksPerStream - (tracksPerStream * nStreams - nTracks);
  Size streamSize =
      (iStream == nStreams - 1) ? tracksLastStream : tracksPerStream;
  std::array<Size, 8> fitData;
  fitData[0] = sizeof(Acts::PixelSourceLink) * nSurfaces * streamSize;
  fitData[1] = sizeof(BoundState) * streamSize;
  fitData[2] = sizeof(Acts::LineSurface) * streamSize;
  fitData[3] = sizeof(FitOptionsType) * streamSize;
  fitData[4] = sizeof(TSType) * nSurfaces * streamSize;
  fitData[5] = sizeof(Acts::BoundParameters<Acts::LineSurface>) * streamSize;
  fitData[6] = sizeof(bool) * streamSize;
  fitData[7] = streamSize;
  return fitData;
}
} // namespace FitDataSizeCalculator
