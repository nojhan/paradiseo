#include <eo>
#include <ga/ga.h>

eoValueParam<float> xoverRate(0.6f, "xoverrate", "The crossover rate", 'x');
eoValueParam<float> mutRate(1.0f, "mutationrate", "The mutation rate", 'm');
eoValueParam<unsigned> chromSize(unsigned(10), "chromosomeSize", "The length of the bitstrings", 'n');
eoValueParam<unsigned> popSize(unsigned(20), "PopSize", "Population Size", 'P');

template <class FitT>
eoAlgo<eoBit<FitT> >&  do_make_ga(eoParameterLoader& _parser, eoEvalFunc<eoBit<FitT> >& _eval, eoCheckPoint<eoBit<FitT> >& _checkpoint, eoState& _state)
{
  typedef eoBit<FitT> EOT;

  _parser.processParam(xoverRate, "genetics");
  _parser.processParam(mutRate, "genetics");
  _parser.processParam(chromSize, "initialization");

  eoBitMutation<EOT>* mutOp = new eoBitMutation<EOT>(1. / float(chromSize.value()));
  _state.storeFunctor(mutOp);

  eo1PtBitXover<EOT>* crossOp = new eo1PtBitXover<EOT>;
  _state.storeFunctor(crossOp);

  eoSelectOne<EOT>* select = new eoDetTournamentSelect<EOT>(2);
  _state.storeFunctor(select);

  eoSGA<eoBit<FitT> >* sga = new eoSGA<EOT>(*select, *crossOp, xoverRate.value(), *mutOp, mutRate.value(), _eval, _checkpoint);
  _state.storeFunctor(sga);
  return *sga;
}

template <class FitT>
eoPop<eoBit<FitT> >&  do_init_ga(eoParameterLoader& _parser, eoState& _state, FitT)
{
  typedef eoBit<FitT> EOT;

  _parser.processParam(chromSize, "initialization");
  _parser.processParam(popSize, "initialization");

  eoInitFixedLength<EOT, boolean_generator> init(chromSize.value(), boolean_generator());


  // Let the state handle the memory
  eoPop<EOT>& pop = _state.takeOwnership(eoPop<EOT>());

  _state.registerObject(pop);

  // initialize the population

  pop.append(popSize.value(), init);

  return pop;
}

template <class FitT>
void do_run_ga(eoAlgo<eoBit<FitT> >& _ga, eoPop<eoBit<FitT> >& _pop)
{
  _ga(_pop);
}

/// The following function merely call the templatized do_* functions above

eoAlgo<eoBit<double> >&  make_ga(eoParameterLoader& _parser, eoEvalFunc<eoBit<double> >& _eval, eoCheckPoint<eoBit<double> >& _checkpoint, eoState& _state)
{
  return do_make_ga(_parser, _eval, _checkpoint, _state);
}

eoAlgo<eoBit<eoMinimizingFitness> >&  make_ga(eoParameterLoader& _parser, eoEvalFunc<eoBit<eoMinimizingFitness> >& _eval, eoCheckPoint<eoBit<eoMinimizingFitness> >& _checkpoint, eoState& _state)
{
  return do_make_ga(_parser, _eval, _checkpoint, _state);
}

eoPop<eoBit<double> >&  init_ga(eoParameterLoader& _parser, eoState& _state, double _d)
{
  return do_init_ga(_parser, _state, _d);
}

eoPop<eoBit<eoMinimizingFitness> >&  init_ga(eoParameterLoader& _parser, eoState& _state, eoMinimizingFitness _d)
{
  return do_init_ga(_parser, _state, _d);
}

void run_ga(eoAlgo<eoBit<double> >& _ga, eoPop<eoBit<double> >& _pop)
{
  do_run_ga(_ga, _pop);
}

void run_ga(eoAlgo<eoBit<eoMinimizingFitness> >& _ga, eoPop<eoBit<eoMinimizingFitness> >& _pop)
{
  do_run_ga(_ga, _pop);
}
