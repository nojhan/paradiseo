#include <peo>
#include <es.h>

typedef eoRealParticle < double >Indi;

double f (const Indi & _indi)
{
  double sum;
  sum=_indi[1]-pow(_indi[0],2);
  sum=100*pow(sum,2);
  sum+=pow((1-_indi[0]),2);
  return (-sum);
}

int main( int __argc, char** __argv )
{
	
  peo :: init( __argc, __argv );
  const unsigned int VEC_SIZE = 2;  
  const unsigned int POP_SIZE = 20; 
  const unsigned int NEIGHBORHOOD_SIZE= 6;
  const unsigned int MAX_GEN = 100; 
  const double INIT_POSITION_MIN = -2.0;
  const double INIT_POSITION_MAX = 2.0; 
  const double INIT_VELOCITY_MIN = -1.;
  const double INIT_VELOCITY_MAX = 1.;
  const unsigned int  MIG_FREQ = 10;
  const double C1 = 0.5;
  const double C2 = 2.; 
  rng.reseed (time(0));

// Island model

  RingTopology topologyMig;
  
// First
  eoGenContinue < Indi > genContPara (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  peoEvalFunc<Indi, double, const Indi& > plainEval(f);
  peoParaPopEval< Indi > eval(plainEval);
  eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
  eoUniformGenerator < double >sGen (INIT_VELOCITY_MIN, INIT_VELOCITY_MAX);
  eoVelocityInitFixedLength < Indi > veloRandom (VEC_SIZE, sGen);
  eoFirstIsBestInit < Indi > localInit;
  eoRealVectorBounds bndsFlight(VEC_SIZE,INIT_POSITION_MIN,INIT_POSITION_MAX);
  eoStandardFlight < Indi > flight(bndsFlight);
  eoPop < Indi > pop;
  pop.append (POP_SIZE, random);
  eoLinearTopology<Indi> topology(NEIGHBORHOOD_SIZE);
  eoRealVectorBounds bnds(VEC_SIZE,INIT_VELOCITY_MIN,INIT_VELOCITY_MAX);
  eoStandardVelocity < Indi > velocity (topology,C1,C2,bnds);
  eoInitializer <Indi> init(eval,veloRandom,localInit,topology,pop);

// Island model

  eoPeriodicContinue< Indi > mig_cont( MIG_FREQ );
  peoPSOSelect<Indi> mig_selec(topology);
  eoSelectNumber< Indi > mig_select(mig_selec);
  peoPSOReplacement<Indi> mig_replace;


// Second

  eoGenContinue < Indi > genContPara2 (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara2 (genContPara2);
  eoCheckPoint<Indi> checkpoint2(continuatorPara2);
  peoEvalFunc<Indi, double, const Indi& > plainEval2(f);
  peoParaPopEval< Indi > eval2(plainEval2);
  eoUniformGenerator < double >uGen2 (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random2 (VEC_SIZE, uGen2);
  eoUniformGenerator < double >sGen2 (INIT_VELOCITY_MIN, INIT_VELOCITY_MAX);
  eoVelocityInitFixedLength < Indi > veloRandom2 (VEC_SIZE, sGen2);
  eoFirstIsBestInit < Indi > localInit2;
  eoRealVectorBounds bndsFlight2(VEC_SIZE,INIT_POSITION_MIN,INIT_POSITION_MAX);
  eoStandardFlight < Indi > flight2(bndsFlight2);
  eoPop < Indi > pop2;
  pop2.append (POP_SIZE, random2);
  eoLinearTopology<Indi> topology2(NEIGHBORHOOD_SIZE);
  eoRealVectorBounds bnds2(VEC_SIZE,INIT_VELOCITY_MIN,INIT_VELOCITY_MAX);
  eoStandardVelocity < Indi > velocity2 (topology2,C1,C2,bnds2);
  eoInitializer <Indi> init2(eval2,veloRandom2,localInit2,topology2,pop2);
  
// Island model

  eoPeriodicContinue< Indi > mig_cont2( MIG_FREQ );
  peoPSOSelect<Indi> mig_selec2(topology2);
  eoSelectNumber< Indi > mig_select2(mig_selec2);
  peoPSOReplacement<Indi> mig_replace2;



// Island model

  peoAsyncIslandMig< Indi > mig( mig_cont, mig_select, mig_replace, topologyMig, pop, pop);
  checkpoint.add( mig );
  peoAsyncIslandMig< Indi > mig2( mig_cont2, mig_select2, mig_replace2, topologyMig, pop2, pop2);
  checkpoint2.add( mig2 );
  

// Parallel algorithm

  eoSyncEasyPSO <Indi> psa(init,checkpoint,eval, velocity, flight);
  peoParallelAlgorithmWrapper parallelPSO( psa, pop);
  eval.setOwner(parallelPSO);
  mig.setOwner(parallelPSO);
  eoSyncEasyPSO <Indi> psa2(init2,checkpoint2,eval2, velocity2, flight2);
  peoParallelAlgorithmWrapper parallelPSO2( psa2, pop2);
  eval2.setOwner(parallelPSO2);
  mig2.setOwner(parallelPSO2);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
  {
    std::cout << "Final population :\n" << pop << std::endl;
    std::cout << "Final population :\n" << pop2	 << std::endl;
  }
}
