add_executable(main_cuda main.cu)
# target_include_directories(main_cuda PRIVATE ${CUDA_TOOLKIT_INCLUDE})
target_compile_features(main_cuda PUBLIC cuda_std_17)
set_target_properties(main_cuda PROPERTIES CUDA_ARCHITECTURES 70-real)
target_link_libraries(main_cuda Actscore)

install(TARGETS main_cuda
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

