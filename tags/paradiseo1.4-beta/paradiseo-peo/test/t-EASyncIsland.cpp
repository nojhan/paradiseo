/*
* <t-EASyncIsland.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Clive Canape
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* peoData to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/


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
  eoPop < Indi > pop;
  pop.append (10, random);
  eoPlusReplacement<Indi> replace;
  eoRandomSelect<Indi> mig_select_one;
  eoSelector <Indi, eoPop<Indi> > mig_select (mig_select_one,2,pop);
  eoReplace <Indi, eoPop<Indi> > mig_replace (replace,pop);
  peoSyncIslandMig<eoPop<Indi>, eoPop<Indi> > mig(2,mig_select,mig_replace,topology);
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
  eoPop < Indi > pop2;
  pop2.append (10, random2);
  eoPlusReplacement<Indi> replace2;
  eoRandomSelect<Indi> mig_select_one2;
  eoSelector <Indi, eoPop<Indi> > mig_select2 (mig_select_one2,2,pop2);
  eoReplace <Indi, eoPop<Indi> > mig_replace2 (replace2,pop2);
  peoSyncIslandMig<eoPop<Indi>, eoPop<Indi> > mig2(2,mig_select2,mig_replace2,topology);
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
