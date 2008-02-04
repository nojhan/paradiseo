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

// Communications between 2 EA and 1 archive


int main (int __argc, char *__argv[])
{

  peo :: init( __argc, __argv );
  const unsigned int VEC_SIZE = 2;
  const unsigned int POP_SIZE = 20;
  const unsigned int MAX_GEN = 100;
  const double INIT_POSITION_MIN = -2.0;
  const double INIT_POSITION_MAX = 2.0;
  const float CROSS_RATE = 0.8;
  const double EPSILON = 0.01;
  const float MUT_RATE = 0.3;
  const unsigned int  MIG_FREQ = 2;
  const unsigned int  MIG_SIZE = 2;
  rng.reseed (time(0));


// Archive

  peoEvalFunc<Indi > plainEval(f);
  eoEvalFuncCounter < Indi > firstEval(plainEval);
  eoPopLoopEval < Indi > evalArchive(firstEval);
  eoUniformGenerator < double > uGenArchive (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > randomArchive (VEC_SIZE, uGenArchive);
  peoPop < Indi > empty_pop,archive;
  archive.append(POP_SIZE, randomArchive);
  evalArchive (empty_pop,archive);
  archive.sort();
  if (getNodeRank()==1)
  	std::cout << "Archive before :\n" << archive << std::endl;
  
  
 
// First algorithm
  /*****************************************************************************************/

  RingTopology topology;

  eoGenContinue < Indi > genContPara (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  peoEvalFunc<Indi> mainEval( f );
  peoPopEval <Indi> eval(mainEval);
  eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
  eoBestSelect<Indi> selectionStrategy;
  eoSelectNumber<Indi> select(selectionStrategy,POP_SIZE);
  eoSegmentCrossover<Indi> crossover;
  eoUniformMutation<Indi>  mutation(EPSILON);
  peoTransform<Indi> transform(crossover,CROSS_RATE,mutation,MUT_RATE);
  peoPop < Indi > pop;
  pop.append (POP_SIZE, random);
  eoPlusReplacement<Indi> replace;
  eoBestSelect <Indi> mig_select_one;
  eoSelector <Indi, peoPop<Indi> > mig_select (mig_select_one,MIG_SIZE,pop);
  eoReplace <Indi, peoPop<Indi> > mig_replace (replace,pop);
  eoPeriodicContinue< Indi > mig_cont( MIG_FREQ );
  eoContinuator<Indi> cont(mig_cont, pop);
  peoAsyncIslandMig< peoPop<Indi>, peoPop<Indi> > mig(cont,mig_select, mig_replace, topology);
  checkpoint.add(mig);

 
  eoRandomSelect<Indi> mig_select_oneArchive;
  eoSelector <Indi, peoPop<Indi> > mig_selectArchive (mig_select_oneArchive,MIG_SIZE,archive);
  eoReplace <Indi, peoPop<Indi> > mig_replaceArchive (replace,archive);
  eoPeriodicContinue< Indi > mig_contArchive( MIG_FREQ );
  eoContinuator<Indi> contArchive(mig_contArchive, pop);
  peoAsyncIslandMig< peoPop<Indi>, peoPop<Indi> > migArchive(contArchive,mig_selectArchive, mig_replaceArchive, topology);
  checkpoint.add(migArchive);
  
  eoEasyEA< Indi > eaAlg( checkpoint, eval, select, transform, replace );
  peoWrapper parallelEA( eaAlg, pop);
  eval.setOwner(parallelEA);
  transform.setOwner(parallelEA);
  mig.setOwner(parallelEA);
  migArchive.setOwner(parallelEA);
  
// Second algorithm
/*****************************************************************************************/

  eoGenContinue < Indi > genContPara2 (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara2 (genContPara2);
  eoCheckPoint<Indi> checkpoint2(continuatorPara2);
  peoEvalFunc<Indi> mainEval2( f );
  peoPopEval <Indi> eval2(mainEval2);
  eoUniformGenerator < double >uGen2 (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random2 (VEC_SIZE, uGen2);
  eoBestSelect<Indi> selectionStrategy2;
  eoSelectNumber<Indi> select2(selectionStrategy2,POP_SIZE);
  eoSegmentCrossover<Indi> crossover2;
  eoUniformMutation<Indi>  mutation2(EPSILON);
  peoTransform<Indi> transform2(crossover2,CROSS_RATE,mutation2,MUT_RATE);
  peoPop < Indi > pop2;
  pop2.append (POP_SIZE, random2);
  eoPlusReplacement<Indi> replace2;
  eoBestSelect <Indi> mig_select_one2;
  eoSelector <Indi, peoPop<Indi> > mig_select2 (mig_select_one2,MIG_SIZE,pop2);
  eoReplace <Indi, peoPop<Indi> > mig_replace2 (replace2,pop2);
  eoPeriodicContinue< Indi > mig_cont2( MIG_FREQ );
  eoContinuator<Indi> cont2(mig_cont2, pop2);
  peoAsyncIslandMig< peoPop<Indi>, peoPop<Indi> > mig2(cont2,mig_select2, mig_replace2, topology);
  checkpoint2.add(mig2);

 
  eoRandomSelect<Indi> mig_select_oneArchive2;
  eoSelector <Indi, peoPop<Indi> > mig_selectArchive2 (mig_select_oneArchive2,MIG_SIZE,archive);
  eoReplace <Indi, peoPop<Indi> > mig_replaceArchive2 (replace2,archive);
  eoPeriodicContinue< Indi > mig_contArchive2( MIG_FREQ );
  eoContinuator<Indi> contArchive2(mig_contArchive2, pop2);
  peoAsyncIslandMig< peoPop<Indi>, peoPop<Indi> > migArchive2(contArchive2,mig_selectArchive2, mig_replaceArchive2, topology);
  checkpoint2.add(migArchive2);
  
  
  
  eoEasyEA< Indi > eaAlg2( checkpoint2, eval2, select2, transform2, replace2 );
  peoWrapper parallelEA2( eaAlg2, pop2);
  eval2.setOwner(parallelEA2);
  transform2.setOwner(parallelEA2);
  mig2.setOwner(parallelEA2);
  migArchive2.setOwner(parallelEA2);


  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
    {
      pop.sort();
      pop2.sort();
      archive.sort();
      std::cout << "Final population 1 :\n" << pop << std::endl;
      std::cout << "Final population 2 :\n" << pop2 << std::endl;
      std::cout << "Archive after :\n" << archive << std::endl;
    }
}


