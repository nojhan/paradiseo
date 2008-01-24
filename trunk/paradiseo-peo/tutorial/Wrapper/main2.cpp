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
  const unsigned  MIG_FREQ = 10;
  const unsigned int  MIG_SIZE = 5;
  rng.reseed (time(0));
  RingTopology topology;
  
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
  peoPop < Indi > pop;
  pop.append (POP_SIZE, random); 
  eoPlusReplacement<Indi> replace;
  eoRandomSelect<Indi> mig_select_one;
  eoSelector <Indi, peoPop<Indi> > mig_select (mig_select_one,MIG_SIZE,pop);
  eoReplace <Indi, peoPop<Indi> > mig_replace (replace,pop);
  peoSyncIslandMig<Indi, peoPop<Indi> > mig(MIG_FREQ,mig_select,mig_replace,topology,pop,pop);
  checkpoint.add(mig);
  eoEasyEA< Indi > eaAlg( checkpoint, eval, select, transform, replace );
  
// Algorithm
  eoGenContinue < Indi > genContPara2 (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara2 (genContPara2);
  eoCheckPoint<Indi> checkpoint2(continuatorPara2);
  peoEvalFunc<Indi> mainEval2( f );
  peoPopEval <Indi> eval2(mainEval2);
  eoUniformGenerator < double >uGen2 (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random2 (VEC_SIZE, uGen2);
  eoRankingSelect<Indi> selectionStrategy2;
  eoSelectNumber<Indi> select2(selectionStrategy2,POP_SIZE);
  eoSegmentCrossover<Indi> crossover2;
  eoUniformMutation<Indi>  mutation2(EPSILON);  
  peoTransform<Indi> transform2(crossover2,CROSS_RATE,mutation2,MUT_RATE); 
  peoPop < Indi > pop2;
  pop2.append (POP_SIZE, random2); 
  eoPlusReplacement<Indi> replace2;
  eoRandomSelect<Indi> mig_select_one2;
  eoSelector <Indi, peoPop<Indi> > mig_select2 (mig_select_one2,MIG_SIZE,pop2);
  eoReplace <Indi, peoPop<Indi> > mig_replace2 (replace2,pop2);
  peoSyncIslandMig<Indi, peoPop<Indi> > mig2(MIG_FREQ,mig_select2,mig_replace2,topology,pop2,pop2);
  checkpoint2.add(mig2);
  eoEasyEA< Indi > eaAlg2( checkpoint2, eval2, select2, transform2, replace2 ); 

  
  peoWrapper parallelEA( eaAlg, pop);
  eval.setOwner(parallelEA);
  transform.setOwner(parallelEA);
  mig.setOwner(parallelEA);
  
  peoWrapper parallelEA2( eaAlg2, pop2);
  eval2.setOwner(parallelEA2);
  transform2.setOwner(parallelEA2);
  mig2.setOwner(parallelEA2);
  
  peo :: run();
  peo :: finalize();


  
 if (getNodeRank()==1)
 {
 	pop.sort();
    std::cout << "Final population :\n" << pop << std::endl;
 }
}
