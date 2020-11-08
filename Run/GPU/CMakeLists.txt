add_executable(KalmanFitterGPUTest KalmanFitterGPUTest.cu)
target_link_libraries(KalmanFitterGPUTest Actscore TestCommon)

install(TARGETS KalmanFitterGPUTest
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

add_executable(IntersectGPUTest IntersectGPUTest.cu)
target_link_libraries(IntersectGPUTest Actscore TestCommon )

install(TARGETS IntersectGPUTest
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

