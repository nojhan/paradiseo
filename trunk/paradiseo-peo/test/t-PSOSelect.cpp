//Test : PSO select
#include <peo>
typedef eoRealParticle < double >Indi;
double f (const Indi & _indi)
{
  double sum=_indi[0]+_indi[1];
  return (-sum);
}

int main (int __argc, char *__argv[])
{
  std::cout<<"\n\nTest : PSO select\n\n";
  rng.reseed (10);
  peoEvalFunc<Indi, double, const Indi& > plainEval(f);
  eoEvalFuncCounter < Indi > firstEval(plainEval);
  eoPopLoopEval < Indi > eval(firstEval);
  eoUniformGenerator < double >uGen (1, 2);
  eoInitFixedLength < Indi > random (2, uGen);
  eoUniformGenerator < double >sGen (-1, 1);
  eoVelocityInitFixedLength < Indi > veloRandom (2, sGen);
  eoFirstIsBestInit < Indi > localInit;
  eoRealVectorBounds bndsFlight(2,1,2);
  eoStandardFlight < Indi > flight(bndsFlight);
  eoLinearTopology<Indi> topology(6);
  eoRealVectorBounds bnds(2,-1,1);
  eoStandardVelocity < Indi > velocity (topology,1,0.5,2.,bnds);
  eoPop < Indi > empty_pop,pop(20, random);
  eoInitializer <Indi> init(eval,veloRandom,localInit,topology,pop);
  init();
  eval (empty_pop,pop);
  peoPSOSelect<Indi> mig_selec(topology);
  pop.sort();
  std::cout<<"\nBest : "<<pop[0]<<"    =     "<<mig_selec(pop)<<"\n\n";
}
