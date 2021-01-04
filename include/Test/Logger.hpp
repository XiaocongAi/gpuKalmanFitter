#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Utility functions to log execution time(s) in an output CSV file
namespace Test {

class Logger {

public:
  // Open/Create the filename with the given 'filename' and
  // print the times separated by commas, on a single line
  template <class... T>
  static void logTime(std::string filename, const T &... times) {

    std::vector<ActsScalar> timesVec = {times...};
    std::ofstream myFile;
    myFile.open(filename, std::ofstream::out | std::ofstream::app);

    for (auto time : timesVec)
      myFile << time << ",";
    myFile << std::endl;

    myFile.close();
  }

  // Concatenate the given 'keywords' separated by '_'
  // e.g. buildFilename("nTracks", "100") -> "nTracks_100";
  // 	buildFilename("nTracks", "100", "1GeV", "CPU") ->
  // "nTracks_100_1GeV_CPU";
  template <typename... T>
  static std::string buildFilename(const T &... keywords) {
    std::vector<std::string> keysVec = {keywords...};
    std::string filename("Results_");
    auto appFn = [&filename](std::string s) { filename.append(s).append("_"); };

    for (auto key : keysVec) {
      appFn(key);
    }

    filename = filename.substr(0, filename.length() - 1).append(".csv");
    return filename;
  }
};

} // end namespace Test
