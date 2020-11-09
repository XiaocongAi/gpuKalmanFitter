add_executable(KalmanFitterGPUTest KalmanFitterGPUTest.cu)
target_link_libraries(KalmanFitterGPUTest Actscore)

target_include_directories(
  KalmanFitterGPUTest 
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../Common>
)


install(TARGETS KalmanFitterGPUTest
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

add_executable(IntersectGPUTest IntersectGPUTest.cu)
target_link_libraries(IntersectGPUTest Actscore)

install(TARGETS IntersectGPUTest
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)
