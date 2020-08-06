using namespace Acts;

template <typename S, typename N>
template <typename parameters_t, typename propagator_options_t,
          typename path_aborter_t>
ACTS_DEVICE_FUNC
    PropagatorResult<CurvilinearParameters,
                     typename propagator_options_t::action_type::result_type>
    Acts::Propagator<S, N>::propagate(
        const parameters_t &start, const propagator_options_t &options) const {
  using ResultType =
      PropagatorResult<CurvilinearParameters,
                       typename propagator_options_t::action_type::result_type>;
  ResultType result;

  using StateType = State<propagator_options_t>;
  StateType state(start, options);

  path_aborter_t pathAborter;
  pathAborter.internalLimit = options.pathLimit;

  // Navigator initialize state call
  m_navigator.status(state, m_stepper);
  // Pre-Stepping call to the action list
  state.options.action(state, m_stepper, result.result);
  // assume negative outcome, only set to true later if we actually have
  // a positive outcome.
  // This is needed for correct error logging
  bool terminatedNormally = false;

  // Pre-Stepping: abort condition check
  if (!state.options.aborter(result, state, m_stepper) and
      !pathAborter(state, m_stepper)) {
    // Pre-Stepping: target setting
    m_navigator.target(state, m_stepper);
    // Propagation loop : stepping
    for (; result.steps < state.options.maxSteps; ++result.steps) {
      // Perform a propagation step - it takes the propagation state
      bool res = m_stepper.step(state);
      // How to handle the error here
      // if (not res) {
      //}
      // Accumulate the path length
      // double s = *res;
      // result.pathLength += s;

      // Post-stepping:
      // navigator status call - action list - aborter list - target call
      m_navigator.status(state, m_stepper);
      state.options.action(state, m_stepper, result.result);
      if (state.options.aborter(result, state, m_stepper) or
          pathAborter(state, m_stepper)) {
        terminatedNormally = true;
        break;
      }
      m_navigator.target(state, m_stepper);
    }
  }

  // if we didn't terminate normally (via aborters) set navigation break.
  // this will trigger error output in the lines below
  if (!terminatedNormally) {
    state.navigation.navigationBreak = true;
  }

  // Post-stepping call to the action list
  state.options.action(state, m_stepper, result.result);

  /// Convert into return type and fill the result object
  auto curvState = m_stepper.curvilinearState(state.stepping);
  // Fill the end parameters
  // result.endParameters = std::make_unique<const
  // CurvilinearParameters>(std::get<CurvilinearParameters>(curvState));
  // Only fill the transport jacobian when covariance transport was done
  if (state.stepping.covTransport) {
    result.transportJacobian = std::get<Jacobian>(curvState);
  }

  return result;
}
