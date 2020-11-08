add_executable(KalmanFitterCPUTest KalmanFitterCPUTest.cpp)
target_link_libraries(KalmanFitterCPUTest Actscore)

target_include_directories(
  KalmanFitterCPUTest 
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../Common>
)

install(TARGETS KalmanFitterCPUTest
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

