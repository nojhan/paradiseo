/*
* <t-ParallelTransform.cpp>
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


// Test : parallel transform
#include <peo>
#include <es.h>
typedef eoReal<double> Indi;
double f (const Indi & _indi)
{
  double sum=_indi[0]+_indi[1];
  return (-sum);
}
struct Algorithm
  {
    Algorithm( eoEvalFunc < Indi > & _eval, eoSelect < Indi > & _select, peoTransform < Indi > & _transform):
        loopEval(_eval),
        eval(loopEval),
        selectTransform( _select, _transform),
        breed(selectTransform) {}

    void operator()(eoPop < Indi > & _pop)
    {
      eoPop < Indi > offspring, empty_pop;
      eval(empty_pop, _pop);
      eval(empty_pop, offspring);
      std::cout<<"\n\nBefore :\n"<<offspring;
      breed(_pop, offspring);
      eval(empty_pop, offspring);
      std::cout<<"\n\nAfter :\n"<<offspring;
    }
    eoPopLoopEval < Indi > loopEval;
    eoPopEvalFunc < Indi > & eval;
    eoSelectTransform < Indi > selectTransform;
    eoBreed < Indi > & breed;
  };

int main (int __argc, char *__argv[])
{
  peo :: init( __argc, __argv );
  if (getNodeRank()==1)
    std::cout<<"\n\nTest : parallel transform\n\n";
  rng.reseed (10);
  eoEvalFuncPtr < Indi > plainEval(f);
  eoEvalFuncCounter < Indi > eval(plainEval);
  eoUniformGenerator < double >uGen (0, 1);
  eoInitFixedLength < Indi > random (2, uGen);
  eoPop < Indi > empty_pop,pop(6, random);
  eoRankingSelect < Indi > selectionStrategy;
  eoSelectNumber < Indi > select(selectionStrategy,6);
  eoSegmentCrossover < Indi > crossover;
  eoUniformMutation < Indi >  mutation(0.01);
  peoTransform<Indi> transform(crossover,0.8,mutation,0.3);
  Algorithm algo ( eval, select, transform );
  peoWrapper parallelAlgo( algo, pop);
  transform.setOwner(parallelAlgo);
  peo :: run();
  peo :: finalize();
}
