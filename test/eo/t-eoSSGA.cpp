#include <paradiseo/eo.h>

// tests a Steady State GA

// Needed to define this breeder, maybe make it a breeder
template <class EOT>
class eoBreedOne : public eoBreed<EOT>
{
public :
  eoBreedOne(eoSelectOne<EOT>& _select, eoGenOp<EOT>& _op) : select(_select), op(_op) {}

  void operator()(const eoPop<EOT>& _src, eoPop<EOT>& _dest)
  {
    _dest.clear();
    eoSelectivePopulator<EOT> pop(_src, _dest, select);
    op(pop);
  }

private :
  eoSelectOne<EOT>& select;
  eoGenOp<EOT>& op;
};

typedef eoMinimizingFitness FitnessType;
typedef eoVector<FitnessType, unsigned> EoType;

template <class EOT>
class eoMyEval : public eoEvalFunc<EOT>
{
  public :

  void operator()(EOT& _eo)
  {
    _eo.fitness(*std::max_element(_eo.begin(), _eo.end()));
  }
};

template <class EOT>
class Xover : public eoBinOp<EOT>
{
  bool operator()(EOT& _eo, const EOT& _eo2)
  {
    unsigned point = rng.random(_eo.size());
    std::copy(_eo2.begin() + point, _eo2.end(), _eo.begin() + point);
    return true;
  }
};

template <class EOT>
class Mutate : public eoMonOp<EOT>
{
  bool operator()(EOT& _eo)
  {
    unsigned point = rng.random(_eo.size());
    _eo[point] = rng.random(1024);
    return true;
  }
};


int main()
{
  int pop_size = 10;

  eoGenContinue<EoType> cnt(10);
  eoCheckPoint<EoType> cp(cnt);


  Xover<EoType> xover;
  Mutate<EoType> mutate;

  eoProportionalOp<EoType> opsel;

  opsel.add(xover, 0.8);
  opsel.add(mutate, 0.2);


  eoDetTournamentSelect<EoType> selector(3);
  eoBreedOne<EoType> breed(selector, opsel);

  // Replace a single one
  eoSSGAWorseReplacement<EoType> replace;


//  eoRandomSelect<EoType> selector;
//  eoGeneralBreeder<EoType> breed(selector, opsel);
//  eoPlusReplacement<EoType> replace;


  eoMyEval<EoType> eval;

  eoEasyEA<EoType> algo(cp, eval, breed, replace);

  eoUniformGenerator<unsigned> unif(0,1024);
  eoInitFixedLength<EoType> init(20, unif);

  eoPop<EoType> pop(pop_size, init);

  // evaluate
  apply<EoType>(eval, pop);

  eoBestFitnessStat<EoType>  best("Best_Fitness");
	eoAverageStat<EoType> avg("Avg_Fitness");
  eoStdoutMonitor mon;

  cp.add(best);
  cp.add(avg);

//  cp.add(mon);

  mon.add(best);
  mon.add(avg);

  // and run
  algo(pop);

}
