
#include <eo>

//#include <utils/eoMOFitnessStat.h>
#include <eoNDSorting.h>
#include <eoParetoFitness.h>

using namespace std;

// Look: overloading the maximization without overhead (thing can be inlined)
class MinimizingFitnessTraits : public eoParetoFitnessTraits
{
  public :
  static bool maximizing(int) { return false; }
};

typedef eoParetoFitness<MinimizingFitnessTraits> fitness_type;

const unsigned chromsize=3;
const double minval = -5;
const double maxval = 5;

struct eoDouble : public EO<fitness_type>
{
  double value[chromsize];
};

class Mutate : public eoMonOp<eoDouble>
{
  bool operator()(eoDouble& _eo)
  {
    for (unsigned i = 0; i < chromsize; ++i)
    {
      if (rng.flip(1./chromsize))
        _eo.value[i] += rng.normal() * 0.1 * _eo.value[i];

      if (_eo.value[i] < minval)
        _eo.value[i] = minval;
      else if (_eo.value[i] > maxval)
        _eo.value[i] = maxval;
    }

    return true;
  }
};

class Eval : public eoEvalFunc<eoDouble>
{
  void operator()(eoDouble& _eo)
  {
    vector<double> x(_eo.value, _eo.value + chromsize);
    fitness_type f;

    for (unsigned i = 0; i < chromsize; ++i)
    {
      if (i < chromsize-1)
      {
        f[0] += -10.0 * exp(-0.2 * sqrt(x[i]*x[i] + x[i+1]*x[i+1]));
      }

      f[1] += pow(fabs(x[i]), 0.8) + 5 * pow(sin(x[i]),3.);
    }

    _eo.fitness(f);
  }
};

class Init : public eoInit<eoDouble>
{
  void operator()(eoDouble& _eo)
  {
    _eo.value[0] = rng.uniform();

    double range = maxval - minval;

    for (unsigned i = 1; i < chromsize; ++i)
      _eo.value[i] = rng.uniform() * range + minval;
    _eo.invalidate();
  }
};

/** @brief An elitist non-dominated sorted replacement scheme.

Trying out an elitist non-dominated sorted replacement scheme.
*/
template <class EOT, class WorthT = double>
class eoNDPlusReplacement : public eoReplacement<EOT>
{
public:

    // using eoNDPlusReplacement< EOT, WorthT >::first;

    eoNDPlusReplacement(eoPerf2Worth<EOT, WorthT>& _perf2worth)
        : perf2worth(_perf2worth)
        {}

    struct WorthPair : public pair<WorthT, const EOT*>
    {
        bool operator<(const WorthPair& other) const
            { return other.first < this->first; }
    };


  void operator()(eoPop<EOT>& _parents, eoPop<EOT>& _offspring)
  {
    unsigned sz = _parents.size();
    _parents.reserve(_parents.size() + _offspring.size());
    std::copy(_offspring.begin(), _offspring.end(), back_inserter(_parents));

    // calculate worths
    perf2worth(_parents);
    perf2worth.sort_pop(_parents);
    perf2worth.resize(_parents, sz);

    _offspring.clear();
  }

private :
    eoPerf2Worth<EOT, WorthT>& perf2worth;
};

template <class EOT>
eoPerf2Worth<EOT, double>& make_perf2worth(eoParser& parser, eoState& state)
{

  unsigned what = parser.createParam(unsigned(1), "perf2worth", "worth mapping indicator : \n\t \
                  0: non_dominated sorting \n\t\
                  1: non_dominated sorting 2 \n\t\
                  2: simple ranking \n\t", 'w').value();

  switch (what)
  {
    case 1 : return state.storeFunctor(new eoNDSorting_II<EOT>());
    case 2 :
    {
      eoDominanceMap<eoDouble>&  dominance = state.storeFunctor(new eoDominanceMap<EOT>);
      return state.storeFunctor(new eoParetoRanking<EOT>(dominance));
    }
  }
  //default

  if (what > 2)
  {
    std::cout << "Warning, need an integer < 3 for perf2worth" << std::endl;
    // should actually set parser flag, but I don't care
  }

  return state.storeFunctor(new eoNDSorting_I<EOT>(0.5));
}

template <class EOT>
eoSelectOne<EOT>& make_selector(eoParser& parser, eoState& state, eoPerf2Worth<EOT, double>& perf2worth)
{
  unsigned tournamentsize = parser.createParam(unsigned(2), "tournament_size", "Tournament Size", 't').value();
  double stochtour = parser.createParam(unsigned(0.95), "tournament_prob", "Probability in stochastic tournament").value();

  switch (parser.createParam(unsigned(0), "selector", "Which selector (too lazy to explain: use the source)", 's').value())
  {
    case 1 : return state.storeFunctor(new eoStochTournamentWorthSelect<eoDouble>(perf2worth, stochtour));
    case 2 : return state.storeFunctor(new eoRouletteWorthSelect<eoDouble>(perf2worth));
    case 3 : return state.storeFunctor(new eoRandomSelect<EOT>);
  }
  // default

  return state.storeFunctor(new eoDetTournamentWorthSelect<eoDouble>(perf2worth, tournamentsize));
}

// Test pareto dominance and perf2worth, and while you're at it, test the eoGnuPlot monitor as well
void the_main(int argc, char* argv[])
{
  Init init;
  Eval eval;
  Mutate mutate;

  eoParser parser(argc, argv);
  eoState state;

  unsigned num_gen  = parser.createParam(unsigned(50), "num_gen", "number of generations to run", 'g').value();
  unsigned pop_size = parser.createParam(unsigned(100), "pop_size", "population size", 'p').value();
  eoPop<eoDouble> pop(pop_size, init);

  // Look, a factory function
  eoPerf2Worth<eoDouble, double>& perf2worth = make_perf2worth<eoDouble>(parser, state);

  // Look: another factory function, now for selection
  eoSelectOne<eoDouble>& select = make_selector<eoDouble>(parser, state, perf2worth);

  // One general operator
  eoProportionalOp<eoDouble> opsel;
  opsel.add(mutate, 1.0);

  // the breeder
  eoGeneralBreeder<eoDouble> breeder(select, opsel);

  // replacement
  eoNDPlusReplacement<eoDouble> replace(perf2worth);

  unsigned long generation = 0;
  eoGenContinue<eoDouble> gen(num_gen, generation);
  eoCheckPoint<eoDouble> cp(gen);

  eoMOFitnessStat<eoDouble> fitness0(0, "FirstObjective");
  eoMOFitnessStat<eoDouble> fitness1(1, "SecondObjective");

  cp.add(fitness0);
  cp.add(fitness1);

#if !defined(NO_GNUPLOT)
  eoGnuplot1DSnapshot snapshot("pareto");
  //snapshot.with(eoGnuplot::Points(3));

  cp.add(snapshot);

  snapshot.add(fitness0);
  snapshot.add(fitness1);
#endif

  // the algo
  eoEasyEA<eoDouble> ea(cp, eval, breeder, replace);

  if (parser.userNeedsHelp())
  {
    parser.printHelp(std::cout);
    return;
  }

  apply<eoDouble>(eval, pop);
  ea(pop);
}


int main(int argc, char* argv[])
{
  try
  {
    the_main(argc, argv);
  }
  catch (std::exception& e)
  {
    std::cout << "Exception thrown: " << e.what() << std::endl;
    throw e; // make sure it does not pass the test
  }
}



// Local Variables:
// mode: C++
// c-file-style: "Stroustrup"
// End:
