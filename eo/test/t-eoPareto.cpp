
#include <eo>

//#include <utils/eoMOFitnessStat.h>
#include <eoNDSorting.h>

using namespace std;
typedef vector<double> fitness_type;

struct eoDouble : public EO<fitness_type>
{
  double value;
};

class Mutate : public eoMonOp<eoDouble>
{
  bool operator()(eoDouble& _eo)
  {
    _eo.value += rng.normal() * 0.1 * _eo.value;
    return true;
  }
};

class Eval : public eoEvalFunc<eoDouble>
{
  void operator()(eoDouble& _eo)
  {
    double v = _eo.value;
    fitness_type f(2);
    f[1] = v * v;
    f[0] = (v - 1.) * (v - 1.);

    _eo.fitness(f);
  }
};

class Init : public eoInit<eoDouble>
{
  void operator()(eoDouble& _eo)
  {
    _eo.value = rng.normal() * 10.;
    _eo.invalidate();
  }
};

// Test pareto dominance and perf2worth, and while you're at it, test the eoGnuPlot monitor as well

void the_main()
{
  Init init;
  Eval eval;
  Mutate mutate;

  unsigned num_gen  = 10;
  unsigned pop_size = 50;
  eoPop<eoDouble> pop(pop_size, init);

  vector<bool> maximizes(2, false); // minimize both objectives

  // The dominance map needs to know how to compare
  eoDominanceMap<eoDouble>  dominance(maximizes);

  // Pareto ranking needs a dominance map
  //eoParetoRanking<eoDouble> perf2worth(dominance);
  eoNDSorting<eoDouble> perf2worth(dominance, 0.0);

  // Three selectors
  eoDetTournamentWorthSelect<eoDouble> select1(perf2worth, 3);
  eoStochTournamentWorthSelect<eoDouble> select2(perf2worth, 0.95);
  eoRouletteWorthSelect<eoDouble> select3(perf2worth);

  // One general operator
  eoProportionalOp<eoDouble> opsel;
  opsel.add(mutate, 1.0);

  // Three breeders
  eoGeneralBreeder<eoDouble> breeder1(select1, opsel);
  eoGeneralBreeder<eoDouble> breeder2(select2, opsel);
  eoGeneralBreeder<eoDouble> breeder3(select3, opsel);

  // Comma replacement
  eoCommaReplacement<eoDouble> replace;

  unsigned long generation = 0;
  eoGenContinue<eoDouble> gen(num_gen, generation);
  eoCheckPoint<eoDouble> cp(gen);

  eoMOFitnessStat<eoDouble> fitness0(0, "FirstObjective");
  eoMOFitnessStat<eoDouble> fitness1(1, "SecondObjective");

  cp.add(fitness0);
  cp.add(fitness1);

  eoGnuplot1DSnapshot snapshot("pareto");
  snapshot.pointSize =3;

  cp.add(snapshot);

  snapshot.add(fitness0);
  snapshot.add(fitness1);

  // Three algos
  eoEasyEA<eoDouble> ea1(cp, eval, breeder1, replace);
  eoEasyEA<eoDouble> ea2(cp, eval, breeder2, replace);
  eoEasyEA<eoDouble> ea3(cp, eval, breeder3, replace);

  apply<eoDouble>(eval, pop);
  ea1(pop);

  apply<eoDouble>(init, pop);
  apply<eoDouble>(eval, pop);
  generation = 0;

  ea2(pop);
  apply<eoDouble>(init, pop);
  apply<eoDouble>(eval, pop);
  generation = 0;

  ea3(pop);

}


int main()
{
  try
  {
    the_main();
  }
  catch (exception& e)
  {
    cout << "Exception thrown: " << e.what() << endl;
    throw e; // make sure it does not pass the test
  }
}

