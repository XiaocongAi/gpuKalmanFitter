add_executable(main_cuda main.cu)

target_compile_features(main_cuda PUBLIC
  cuda_std_17)

target_compile_options(main_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -expt-relaxed-constexpr -expt-extended-lambda -keep -extra-device-vectorization -restrict -Wreorder -res-usage>)

set_target_properties(main_cuda PROPERTIES
  CUDA_ARCHITECTURES 70-real)

target_link_libraries(main_cuda Actscore)

install(TARGETS main_cuda
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin      COMPONENT runtime
  LIBRARY       DESTINATION bin      COMPONENT runtime
  ARCHIVE       DESTINATION bin      COMPONENT devel
  PUBLIC_HEADER DESTINATION include  COMPONENT devel
  RESOURCE      DESTINATION resource COMPONENT runtime
)

