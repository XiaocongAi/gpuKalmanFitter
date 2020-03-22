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
  int iStep = 0;
  for (; iStep < maxSteps; ++iStep) {
    m_stepper.step(state);
    Vector3DMap(result.position.col(iStep).data()) = state.stepping.pos;
    double x = state.stepping.pos.x();
    //  printf("pos.x = %f\n", x);
    // std::cout<<"cout pos.x = "<<state.stepping.pos.x()<<std::endl;
    Vector3DMap(result.momentum.col(iStep).data()) =
        state.stepping.p * state.stepping.dir;
  }
}
