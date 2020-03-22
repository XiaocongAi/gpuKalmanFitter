using namespace Acts;

template <typename S>
template <typename parameters_t, typename propagator_options_t,
          typename result_t>
__host__ __device__ void
Acts::Propagator<S>::propagate(const parameters_t &start,
                               const propagator_options_t &options,
                               result_t &result) const {
  using StateType = State<propagator_options_t>;
  StateType state(start, options);

  int maxSteps = state.options.maxSteps < result.nSteps()
                     ? state.options.maxSteps
                     : result.nSteps();
  //  printf("maxSteps = %f\n", maxSteps);
  int iStep = 0;
  for (; iStep < maxSteps; ++iStep) {
    m_stepper.step(state);
    Vector3DMap(result.position.col(iStep).data()) = state.stepping.pos;
    Vector3DMap(result.momentum.col(iStep).data()) =
        state.stepping.p * state.stepping.dir;
    // printf("pos = (%f, %f, %f)\n", state.stepping.pos.x(),
    // state.stepping.pos.y(), state.stepping.pos.z());
  }
}
