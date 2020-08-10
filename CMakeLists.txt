cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(Actscore VERSION 1.0.0 LANGUAGES CXX CUDA)

set(Eigen3_DIR "/usr/local/share/eigen3/cmake" CACHE STRING "Location of Eigen3 CMake configuration file -- usually INSTALL_PREFIX/share/eigen3/cmake")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/INSTALL CACHE PATH "Default prefix of install directories" FORCE)
endif()

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

#set(CMAKE_CXX_STANDARD 17)

#if(NOT DEFINED CMAKE_CUDA_STANDARD)
#    set(CMAKE_CUDA_STANDARD 17)
#    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#endif()

aux_source_directory(src/MagneticField LIB_SRC_BField)
aux_source_directory(src/Plugins LIB_SRC_Plugins)

# make sure libActscore.so is available at runtime
set(CMAKE_INSTALL_RPATH "$ORIGIN/../bin")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_library(Actscore SHARED ${LIB_SRC_MAT} ${LIB_SRC_SUR} ${LIB_SRC_BField} ${LIB_SRC_Plugins})
target_compile_features(Actscore PUBLIC cxx_std_17)
target_link_libraries(Actscore PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen)

target_include_directories(Actscore
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  )

add_subdirectory(exe_cuda)

# install-tree
set(CONF_INCLUDE_DIRS "\${CMAKE_CURRENT_LIST_DIR}/../include")
configure_file(cmake/ActscoreConfig.cmake.in         ${PROJECT_BINARY_DIR}/cmake_install/ActscoreConfig.cmake @ONLY)
configure_file(cmake/ActscoreConfigVersion.cmake.in  ${PROJECT_BINARY_DIR}/cmake_install/ActscoreConfigVersion.cmake @ONLY)
install(FILES
  ${PROJECT_BINARY_DIR}/cmake_install/ActscoreConfig.cmake
  ${PROJECT_BINARY_DIR}/cmake_install/ActscoreConfigVersion.cmake
  DESTINATION cmake COMPONENT devel)

install(TARGETS Actscore
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime 
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
  )

install(EXPORT ${PROJECT_NAME}Targets
  DESTINATION cmake
  )

install(CODE "MESSAGE(\"project is installed in ${CMAKE_INSTALL_PREFIX} .\")")

# build-tree
set(CONF_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR})
configure_file(cmake/ActscoreConfig.cmake.in          ${PROJECT_BINARY_DIR}/cmake/ActscoreConfig.cmake @ONLY)
configure_file(cmake/ActscoreConfigVersion.cmake.in   ${PROJECT_BINARY_DIR}/cmake/ActscoreConfigVersion.cmake @ONLY)

export(TARGETS Actscore  FILE ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake)
export(PACKAGE ${PROJECT_NAME})
