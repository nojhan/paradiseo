/*
* <t-ParallelEval.cpp>
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


// Test : parallel evaluation
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
    Algorithm( peoPopEval < Indi > & _eval): eval( _eval ){}
    void operator()(eoPop < Indi > & _pop)
    {
      eoPop < Indi > empty_pop;
      eval(empty_pop, _pop);
    }
    peoPopEval < Indi > & eval;
  };

int main (int __argc, char *__argv[])
{
  peo :: init( __argc, __argv );
  if (getNodeRank()==1)
    std::cout<<"\n\nTest : parallel evaluation\n\n";
  rng.reseed (10);
  peoEvalFunc<Indi, double, const Indi& > plainEval(f);
  peoPopEval< Indi > eval(plainEval);
  eoUniformGenerator < double >uGen (0, 1);
  eoInitFixedLength < Indi > random (2, uGen);
  eoPop < Indi > pop(20, random);
  Algorithm algo ( eval );
  peoWrapper parallelAlgo( algo, pop);
  eval.setOwner(parallelAlgo);
  peo :: run();
  peo :: finalize();
  if (getNodeRank()==1)
    {
      pop.sort();
      std::cout<<pop;
    }
}
