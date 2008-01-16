#include <peo>
#include <es.h>
typedef eoReal<double> Indi;

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

// Parameters
  const unsigned int VEC_SIZE = 2;  
  const unsigned int POP_SIZE = 20; 
  const unsigned int MAX_GEN = 300; 
  const double INIT_POSITION_MIN = -2.0;  
  const double INIT_POSITION_MAX = 2.0;   
  const float CROSS_RATE = 0.8; 
  const double EPSILON = 0.01; 
  const float MUT_RATE = 0.5;  
  rng.reseed (time(0));

// Algorithm
  eoGenContinue < Indi > genContPara (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  
  peoEvalFunc<Indi> mainEval( f );
  peoPopEval <Indi> eval(mainEval);
  
  eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
  eoRankingSelect<Indi> selectionStrategy;
  eoSelectNumber<Indi> select(selectionStrategy,POP_SIZE);
  eoSegmentCrossover<Indi> crossover;
  eoUniformMutation<Indi>  mutation(EPSILON);  
  
  peoTransform<Indi> transform(crossover,CROSS_RATE,mutation,MUT_RATE);
    
  eoPlusReplacement<Indi> replace;
  eoEasyEA< Indi > eaAlg( checkpoint, eval, select, transform, replace );  

// Population*/
  eoPop < Indi > pop;
  pop.append (POP_SIZE, random);
  peoWrapper parallelEA( eaAlg, pop);
  eval.setOwner(parallelEA);
  transform.setOwner(parallelEA);
  peo :: run();
  peo :: finalize();


  
 if (getNodeRank()==1)
    std::cout << "Final population :\n" << pop << std::endl;
}
