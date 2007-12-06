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
  const float MUT_RATE = 0.3;  
  rng.reseed (time(0));


// Algorithm
  eoGenContinue < Indi > genContPara (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  eoEvalFuncPtr< Indi > mainEval( f );
  eoEvalFuncCounter< Indi > eval(mainEval);
  eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
  eoRankingSelect<Indi> selectionStrategy;
  eoSelectNumber<Indi> select(selectionStrategy,POP_SIZE);
  eoSegmentCrossover<Indi> crossover;
  eoUniformMutation<Indi>  mutation(EPSILON);  
  eoSGATransform<Indi> transform(crossover,CROSS_RATE,mutation,MUT_RATE);
  peoSeqTransform<Indi> eaTransform(transform);
  eoPlusReplacement<Indi> replace;
  eoEasyEA< Indi > eaAlg( checkpoint, eval, select, transform, replace );  

// Population
  eoPop < Indi > pop;
  pop.append (POP_SIZE, random);

// Wrapper 
  peoParallelAlgorithmWrapper parallelEA( eaAlg, pop);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
   	std::cout << "Final population :\n" << pop << std::endl;
  

// Algorithm
  eoGenContinue < Indi > genContPara2 (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara2 (genContPara2);
  eoCheckPoint<Indi> checkpoint2(continuatorPara2);
  eoEvalFuncPtr< Indi > mainEval2( f );
  eoEvalFuncCounter< Indi > eval2(mainEval2);
  eoUniformGenerator < double >uGen2 (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random2 (VEC_SIZE, uGen2);
  eoRankingSelect<Indi> selectionStrategy2;
  eoSelectNumber<Indi> select2(selectionStrategy2,POP_SIZE);
  eoSegmentCrossover<Indi> crossover2;
  eoUniformMutation<Indi>  mutation2(EPSILON);  
  eoSGATransform<Indi> transform2(crossover2,CROSS_RATE,mutation2,MUT_RATE);
  peoSeqTransform<Indi> eaTransform2(transform2);
  eoPlusReplacement<Indi> replace2;
  eoEasyEA< Indi > eaAlg2( checkpoint2, eval2, select2, transform2, replace2 );  

// Population
  eoPop < Indi > pop2;
  pop2.append (POP_SIZE, random2);


// Wrapper
  peo :: init( __argc, __argv );
  peoParallelAlgorithmWrapper parallelEA2( eaAlg2, pop2);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
  	std::cout << "Final population 2 :\n" << pop2 << std::endl;
}
