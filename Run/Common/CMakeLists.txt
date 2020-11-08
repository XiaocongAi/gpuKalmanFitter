add_library(TestCommon SHARED Processor.cpp)
target_link_libraries(TestCommon Actscore)

install(TARGETS TestCommon 
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

