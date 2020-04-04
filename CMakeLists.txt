cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(Actscore VERSION 1.0.0 LANGUAGES CXX CUDA)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/INSTALL CACHE PATH "Default prefix of install directories" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

aux_source_directory(src/MagneticField LIB_SRC_BField)
aux_source_directory(src/Plugins LIB_SRC_Plugins)

find_package(Eigen3)

add_library(Actscore SHARED ${LIB_SRC_MAT} ${LIB_SRC_SUR} ${LIB_SRC_BField} ${LIB_SRC_Plugins})
target_link_libraries(Actscore PUBLIC Eigen3::Eigen)
target_include_directories(Actscore
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  )

#add_subdirectory(exe_gcc)
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