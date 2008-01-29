// Test of parallel evaluation with a PSO
#include <peo>
typedef eoRealParticle < double >Indi;
double f (const Indi & _indi)
{
  double sum=_indi[0]+_indi[1];
  return (-sum);
}
int main (int __argc, char *__argv[])
{
  peo :: init( __argc, __argv );
  rng.reseed (time(0));
  eoGenContinue < Indi > genContPara (5);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  peoEvalFunc<Indi, double, const Indi& > plainEval(f);
  peoPopEval< Indi > eval(plainEval);
  eoUniformGenerator < double >uGen (-2., 2.);
  eoInitFixedLength < Indi > random (2, uGen);
  eoUniformGenerator < double >sGen (-1., 1.);
  eoVelocityInitFixedLength < Indi > veloRandom (2, sGen);
  eoFirstIsBestInit < Indi > localInit;
  eoRealVectorBounds bndsFlight(2,-2.,2.);
  eoStandardFlight < Indi > flight(bndsFlight);
  eoPop < Indi > pop;
  pop.append (20, random);
  eoLinearTopology<Indi> topology(6);
  eoRealVectorBounds bnds(2,-1.,1.);
  eoStandardVelocity < Indi > velocity (topology,1,0.5,2.,bnds);
  eoInitializer <Indi> init(eval,veloRandom,localInit,topology,pop);
  eoSyncEasyPSO <Indi> psa(init,checkpoint,eval, velocity, flight);
  peoWrapper parallelPSO( psa, pop);
  eval.setOwner(parallelPSO);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
  {
      pop.sort();
      std::cout<<pop;
  }
}
