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
  const unsigned int MAX_GEN = 300; 
  const double INIT_POSITION_MIN = -2.0;
  const double INIT_POSITION_MAX = 2.0; 
  const double INIT_VELOCITY_MIN = -1.;
  const double INIT_VELOCITY_MAX = 1.;
  const double weight = 1;
  const double C1 = 0.5;
  const double C2 = 2.; 
  rng.reseed (time(0));
  eoGenContinue < Indi > genContPara (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  
  peoEvalFunc<Indi, double, const Indi& > plainEval(f);
  peoPopEval< Indi > eval(plainEval);
  
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
  eoStandardVelocity < Indi > velocity (topology,weight,C1,C2,bnds);
  eoInitializer <Indi> init(eval,veloRandom,localInit,topology,pop);

// Parallel algorithm

  eoSyncEasyPSO <Indi> psa(init,checkpoint,eval, velocity, flight);
  peoWrapper parallelPSO( psa, pop);
  eval.setOwner(parallelPSO);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
  {
  	pop.sort();
    std::cout << "Final population :\n" << pop << std::endl;
  }
}
