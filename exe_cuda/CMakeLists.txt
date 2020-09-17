add_executable(KalmanFitterCudaTest KalmanFitterCudaTest.cu)
target_link_libraries(KalmanFitterCudaTest Actscore )

install(TARGETS KalmanFitterCudaTest
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

add_executable(IntersectCudaTest IntersectCudaTest.cu)
target_link_libraries(IntersectCudaTest Actscore )

install(TARGETS IntersectCudaTest
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

