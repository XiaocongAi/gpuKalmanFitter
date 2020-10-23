add_executable(KalmanFitterGccTest KalmanFitterGccTest.cpp)
target_link_libraries(KalmanFitterGccTest Actscore PerfTesting)

install(TARGETS KalmanFitterGccTest
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

