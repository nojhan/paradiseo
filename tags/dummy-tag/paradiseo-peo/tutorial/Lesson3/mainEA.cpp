/*
* <main.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, INRIA, 2008
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
* with loading,  using,  modifying and/syncor developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

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

int main (int __argc, char *__argv[])
{

  peo :: init( __argc, __argv );
  eoParser parser(__argc, __argv);
  unsigned int POP_SIZE = parser.createParam((unsigned int)(20), "popSize", "Population size",'P',"Param").value();
  unsigned int MAX_GEN = parser.createParam((unsigned int)(100), "maxGen", "Maximum number of generations",'G',"Param").value();
  double EPSILON = parser.createParam(0.01, "mutEpsilon", "epsilon for mutation",'e',"Param").value();
  double CROSS_RATE = parser.createParam(0.25, "pCross", "Crossover probability",'C',"Param").value();
  double MUT_RATE = parser.createParam(0.35, "pMut", "Mutation probability",'M',"Param").value();
  unsigned int VEC_SIZE = parser.createParam((unsigned int)(2), "vecSize", "Vector size",'V',"Param").value();  
  double INIT_POSITION_MIN = parser.createParam(-2.0, "pMin", "Init position min",'N',"Param").value();
  double INIT_POSITION_MAX = parser.createParam(2.0, "pMax", "Init position max",'X',"Param").value();
  unsigned int MIG_FREQ = parser.createParam((unsigned int)(10), "migFreq", "Migration frequency",'F',"Param").value();
  unsigned int MIG_SIZE = parser.createParam((unsigned int)(5), "migSize", "Migration size",'S',"Param").value();
  rng.reseed (time(0));

// Define the topology of your island model
  RingTopology topology;

// First algorithm
  /*****************************************************************************************/

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
  eoPop < Indi > pop;
  pop.append (POP_SIZE, random);
  
// Define a synchronous island
  
  // Seclection
  eoRandomSelect<Indi> mig_select_one;
  eoSelector <Indi, eoPop<Indi> > mig_select (mig_select_one,MIG_SIZE,pop);
  // Replacement
  eoPlusReplacement<Indi> replace;
  eoReplace <Indi, eoPop<Indi> > mig_replace (replace,pop);
  // Island
  peoSyncIslandMig<eoPop<Indi>, eoPop<Indi> > mig(MIG_FREQ,mig_select,mig_replace,topology);
  checkpoint.add(mig);
  
  eoEasyEA< Indi > eaAlg( checkpoint, eval, select, transform, replace );
  peoWrapper parallelEA( eaAlg, pop);
  eval.setOwner(parallelEA);
  transform.setOwner(parallelEA);
// setOwner
  mig.setOwner(parallelEA);

  /*****************************************************************************************/

// Second algorithm (on the same model but with others names)
  /*****************************************************************************************/

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
  eoPop < Indi > pop2;
  pop2.append (POP_SIZE, random2);
  eoPlusReplacement<Indi> replace2;
  eoRandomSelect<Indi> mig_select_one2;
  eoSelector <Indi, eoPop<Indi> > mig_select2 (mig_select_one2,MIG_SIZE,pop2);
  eoReplace <Indi, eoPop<Indi> > mig_replace2 (replace2,pop2);
  peoSyncIslandMig<eoPop<Indi>, eoPop<Indi> > mig2(MIG_FREQ,mig_select2,mig_replace2,topology);
  checkpoint2.add(mig2);
  eoEasyEA< Indi > eaAlg2( checkpoint2, eval2, select2, transform2, replace2 );
  peoWrapper parallelEA2( eaAlg2, pop2);
  eval2.setOwner(parallelEA2);
  transform2.setOwner(parallelEA2);
  mig2.setOwner(parallelEA2);

  /*****************************************************************************************/

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
