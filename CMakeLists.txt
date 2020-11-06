cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(Actscore VERSION 1.0.0 LANGUAGES CXX CUDA)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/INSTALL CACHE PATH "Default prefix of install directories" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# set(CMAKE_CXX_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)

# if(NOT DEFINED CMAKE_CUDA_STANDARD)
#     set(CMAKE_CUDA_STANDARD 17)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif()

aux_source_directory(src/MagneticField LIB_SRC_BField)
aux_source_directory(src/Plugins LIB_SRC_Plugins)
aux_source_directory(src/Material LIB_SRC_Material)

find_package(CUDA)
message(${CUDA_TOOLKIT_INCLUDE})
include_directories(${CUDA_TOOLKIT_INCLUDE})
# include_directories(${CUDA_TOOLKIT_INCLUDE}/crt)
find_package(OpenMP)

# import Eigen
include(cmake/FindEigen.cmake)

# make sure libActscore.so is available at runtime
#set(CMAKE_INSTALL_RPATH "$ORIGIN/../bin")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -std=c++14 -lineinfo -res-usage -arch=sm_70 -Xcompiler -fopenmp") 


add_library(Actscore STATIC ${LIB_SRC_BField} ${LIB_SRC_Plugins} ${LIB_SRC_Material})

target_link_libraries(Actscore PUBLIC OpenMP::OpenMP_CXX  Eigen3::Eigen)
target_include_directories(Actscore
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  )

#add_subdirectory(exe_cuda)
add_subdirectory(exe_gcc)

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

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/Performance/scripts
                                                           ${CMAKE_SOURCE_DIR}/INSTALL/scripts)

install(CODE "MESSAGE(\"project is installed in ${CMAKE_INSTALL_PREFIX} .\")")

# build-tree
set(CONF_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/performance_testing)
configure_file(cmake/ActscoreConfig.cmake.in          ${PROJECT_BINARY_DIR}/cmake/ActscoreConfig.cmake @ONLY)
configure_file(cmake/ActscoreConfigVersion.cmake.in   ${PROJECT_BINARY_DIR}/cmake/ActscoreConfigVersion.cmake @ONLY)

export(TARGETS Actscore  FILE ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake)
export(PACKAGE ${PROJECT_NAME})
