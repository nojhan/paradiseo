/*
* <main.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, INRIA, 2007
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
  const unsigned int VEC_SIZE = 2;
  const unsigned int POP_SIZE = 20;
  const unsigned int MAX_GEN = 300;
  const double INIT_POSITION_MIN = -2.0;
  const double INIT_POSITION_MAX = 2.0;
  const float CROSS_RATE = 0.8;
  const double EPSILON = 0.01;
  const float MUT_RATE = 0.3;
  rng.reseed (time(0));
  eoGenContinue < Indi > genContPara (MAX_GEN);
  eoCombinedContinue <Indi> continuatorPara (genContPara);
  eoCheckPoint<Indi> checkpoint(continuatorPara);
  peoEvalFunc<Indi> plainEval(f);
  peoSeqPopEval< Indi > eval(plainEval);  // Here, the evaluation is sequential
  eoUniformGenerator < double >uGen (INIT_POSITION_MIN, INIT_POSITION_MAX);
  eoInitFixedLength < Indi > random (VEC_SIZE, uGen);
  eoRankingSelect<Indi> selectionStrategy;
  eoSelectNumber<Indi> select(selectionStrategy,POP_SIZE);
  eoSegmentCrossover<Indi> crossover;
  eoUniformMutation<Indi>  mutation(EPSILON);

  /******************************************************************************************/

  /* In this lesson, you can choose between  :
   * 
   *    - A sequential transformation (crossover + mutation) : eoSGATransform<Indi> transform(crossover,CROSS_RATE,mutation,MUT_RATE);
   *                                                           peoSeqTransform<Indi> eaTransform(transform);
   * 
   *   OR
   * 
   *    - A parallel transformation (crossover + mutation) : peoParaSGATransform <Indi> eaTransform(crossover,CROSS_RATE,mutation,MUT_RATE); 
   *
   *  Unfortunately, if you don't use a crossover which creates two children with two parents,
   *  you can't use this operator.
   *  In this case, you should send a mail to : paradiseo-help@lists.gforge.inria.fr 
   */

  peoParaSGATransform <Indi> eaTransform(crossover,CROSS_RATE,mutation,MUT_RATE);

  /******************************************************************************************/

  eoPlusReplacement<Indi> replace;
  eoPop < Indi > pop;
  pop.append (POP_SIZE, random);
  peoEA<Indi> Algo(checkpoint,eval,select,eaTransform,replace);
  Algo(pop);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
    std::cout << "Final population :\n" << pop << std::endl;
}
