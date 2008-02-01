//Test : Synchronous island with EA
#include <peo>
#include <es.h>
typedef eoReal<double> Indi;

double f (const Indi & _indi)
{
  double sum=_indi[0]+_indi[1];
  return (-sum);
}
int main (int __argc, char *__argv[])
{
  peo :: init( __argc, __argv );
  if (getNodeRank()==1)
  	std::cout<<"\n\nTest : Synchronous island with EA\n\n";
  rng.reseed (10);
  RingTopology topology;
  eoGenContinue < Indi > genContPara (10);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  peoEvalFunc<Indi> mainEval( f );
  peoPopEval <Indi> eval(mainEval);
  eoUniformGenerator < double >uGen (-2., 2.);
  eoInitFixedLength < Indi > random (2, uGen);
  eoRankingSelect<Indi> selectionStrategy;
  eoSelectNumber<Indi> select(selectionStrategy,10);
  eoSegmentCrossover<Indi> crossover;
  eoUniformMutation<Indi>  mutation(0.01);
  peoTransform<Indi> transform(crossover,0.8,mutation,0.3);
  peoPop < Indi > pop;
  pop.append (10, random);
  eoPlusReplacement<Indi> replace;
  eoRandomSelect<Indi> mig_select_one;
  eoSelector <Indi, peoPop<Indi> > mig_select (mig_select_one,2,pop);
  eoReplace <Indi, peoPop<Indi> > mig_replace (replace,pop);
  eoPeriodicContinue< Indi > mig_cont( 2 );
  eoContinuator<Indi> cont(mig_cont, pop);
  peoAsyncIslandMig<Indi, peoPop<Indi> > mig(cont,mig_select,mig_replace,topology,pop,pop);
  checkpoint.add(mig);
  eoEasyEA< Indi > eaAlg( checkpoint, eval, select, transform, replace );
  peoWrapper parallelEA( eaAlg, pop);
  eval.setOwner(parallelEA);
  transform.setOwner(parallelEA);
  mig.setOwner(parallelEA);
  eoGenContinue < Indi > genContPara2 (10);
  eoCombinedContinue <Indi> continuatorPara2 (genContPara2);
  eoCheckPoint<Indi> checkpoint2(continuatorPara2);
  peoEvalFunc<Indi> mainEval2( f );
  peoPopEval <Indi> eval2(mainEval2);
  eoUniformGenerator < double >uGen2 (-2., 2.);
  eoInitFixedLength < Indi > random2 (2, uGen2);
  eoRankingSelect<Indi> selectionStrategy2;
  eoSelectNumber<Indi> select2(selectionStrategy2,10);
  eoSegmentCrossover<Indi> crossover2;
  eoUniformMutation<Indi>  mutation2(0.01);
  peoTransform<Indi> transform2(crossover2,0.8,mutation2,0.3);
  peoPop < Indi > pop2;
  pop2.append (10, random2);
  eoPlusReplacement<Indi> replace2;
  eoRandomSelect<Indi> mig_select_one2;
  eoSelector <Indi, peoPop<Indi> > mig_select2 (mig_select_one2,2,pop2);
  eoReplace <Indi, peoPop<Indi> > mig_replace2 (replace2,pop2);
  eoPeriodicContinue< Indi > mig_cont2( 2 );
  eoContinuator<Indi> cont2(mig_cont2, pop2);
  peoAsyncIslandMig<Indi, peoPop<Indi> > mig2(cont2,mig_select2,mig_replace2,topology,pop2,pop2);
  checkpoint2.add(mig2);
  eoEasyEA< Indi > eaAlg2( checkpoint2, eval2, select2, transform2, replace2 );
  peoWrapper parallelEA2( eaAlg2, pop2);
  eval2.setOwner(parallelEA2);
  transform2.setOwner(parallelEA2);
  mig2.setOwner(parallelEA2);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
    {
      pop.sort();
      pop2.sort();
      std::cout << "Final population 1 :\n" << pop << std::endl;
      std::cout << "Final population 2 :\n" << pop2 << std::endl;
    }
}
