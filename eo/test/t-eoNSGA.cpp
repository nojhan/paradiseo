
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <eo>

#include <moo/eoMOFitness.h>
#include <moo/eoNSGA_I_Eval.h>
#include <moo/eoNSGA_II_Eval.h>
#include <moo/eoNSGA_IIa_Eval.h>

using namespace std;

class MinimizingFitnessTraits 
{
  public :
  static unsigned nObjectives()         { return 2; }
  static double direction(unsigned)     { return -1; }
  static double tol()                   { return 1e-9; }
};

typedef eoMOFitness<MinimizingFitnessTraits> fitness_type;

const unsigned chromsize=5;
const double minval = -15;
const double maxval = 15;

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

class Cross : public eoBinOp<eoDouble> {
  
  bool operator()(eoDouble& _eo, const eoDouble& eo2)
  {
    for (unsigned i = 0; i < chromsize; ++i)
    {
      if (rng.flip(0.5)) {
           _eo.value[i] = eo2.value[i];
    }
    }
    return true;
  }

};

class Eval : public eoEvalFunc<eoDouble>
{
  void operator()(eoDouble& _eo)
  {
    vector<double> x(_eo.value, _eo.value + chromsize);
    fitness_type f(0.0);

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

class Eval2 : public eoEvalFunc<eoDouble>
{
  void operator()(eoDouble& _eo)
  {
    vector<double> x(_eo.value, _eo.value + chromsize);
    fitness_type f;
    
    for (unsigned i = 0; i < chromsize; ++i)
    {
      f[0] += x[i] * x[i];
    }
    
    f[1] = 
        3 * x[0] + 2 * x[1] 
        - x[2]/3 + 0.01*pow(x[3] - x[4], 3);

    _eo.fitness(f);
  }
};

class Eval3 : public eoEvalFunc<eoDouble>
{
  void operator()(eoDouble& _eo)
  {
      double x = _eo.value[0];
    fitness_type f;
    
    f[0] = x * x;
    double y = x-10;
    f[1] = y * y;

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

// Test pareto dominance and perf2worth, and while you're at it, test the eoGnuPlot monitor as well
void the_main(int argc, char* argv[])
{
  Init init;
  Eval simple_eval;
  Mutate mutate;
  Cross cross;

  eoParser parser(argc, argv);
  eoState state;

  unsigned num_gen  = parser.createParam(unsigned(50), "num_gen", "number of generations to run", 'g').value();
  unsigned pop_size = parser.createParam(unsigned(100), "pop_size", "population size", 'p').value();
  bool use_nsga1     = parser.createParam(false, "nsga1", "Use nsga 1").value(); 
  bool use_nsga2a     = parser.createParam(false, "nsga2a", "Use nsga 2a").value(); 
  
  eoPop<eoDouble> pop(pop_size, init);

  // binary tournament selection 
  eoDetTournamentSelect<eoDouble> select;

  // One general operator
  eoProportionalOp<eoDouble> opsel;
  opsel.add(mutate, 1.0);
  opsel.add(cross, 1.0);

  // the breeder
  eoGeneralBreeder<eoDouble> breeder(select, opsel);

  // replacement
  eoNSGA_IIa_Eval<eoDouble> nsga2a(simple_eval);
  eoNSGA_II_Eval<eoDouble> nsga2(simple_eval);
  eoNSGA_I_Eval<eoDouble> nsga1(0.1, simple_eval);
  eoMOEval<eoDouble>& eval = use_nsga1 ? static_cast<eoMOEval<eoDouble>&>(nsga1) : (use_nsga2a? static_cast<eoMOEval<eoDouble>&>(nsga2a) : static_cast<eoMOEval<eoDouble>&>(nsga2));

  eoPlusReplacement<eoDouble> replace;

  unsigned long generation = 0;
  eoGenContinue<eoDouble> gen(num_gen, generation);
  eoCheckPoint<eoDouble> cp(gen);

  eoMOFitnessStat<eoDouble> fitness0(0, "FirstObjective");
  eoMOFitnessStat<eoDouble> fitness1(1, "SecondObjective");

  cp.add(fitness0);
  cp.add(fitness1);

#ifdef HAVE_GNUPLOT
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
  
  ea(pop);
}


int main(int argc, char* argv[])
{
    the_main(argc, argv);
    return 0;
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
