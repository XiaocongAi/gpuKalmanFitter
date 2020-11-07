#pragma once

#include <cstring>
#include <cuda.h>
#include <string>
#include <vector>

#define GPUERRCHK(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define DELIMITER "*"

// Convert GPU string configuration from gridSize and/or blockSize to cuda::dim3
//
// Valid configurations:
// 	1. "<num_x>*<num_y>*<num_z>"  ->> use this for more clarity
//	2. "<num_x>*<num_y>"
//	3. "<num_x>"
// OBS: when a dimenssion is missing, default value is 1.
dim3 stringToDim3(char *config) {
  const int defaultVal = 1;
  char *token = std::strtok(config, DELIMITER);
  std::vector<int> tokens;

  while (token != NULL) {
    tokens.push_back(atoi(token));
    token = std::strtok(NULL, DELIMITER);
  }

  for (int i = tokens.size(); i < 3; i++)
    tokens.push_back(defaultVal);

  return dim3(tokens[0], tokens[1], tokens[2]);
}

// Convert cuda::dim3 to string configurations for gridSize and/or blockSize
std::string dim3ToString(dim3 size) {
  std::string result;
  result.append(std::to_string(size.x))
      .append(DELIMITER)
      .append(std::to_string(size.y))
      .append(DELIMITER)
      .append(std::to_string(size.z));
  return result;
}

std::string filename(int nTracks, std::string gridSize, std::string blockSize) {
  std::string filename;
  filename.append("nTracks_")
      .append(std::to_string(nTracks))
      .append("_gridSize_")
      .append(gridSize)
      .append("_blockSize_")
      .append(blockSize)
      .append(".csv");
  return filename;
}
