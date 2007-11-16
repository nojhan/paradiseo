/*
* <Sch1.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Abdelhakim Deneche
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
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <moeo>
#include <es/eoRealInitBounded.h>
#include <es/eoRealOp.h>

using namespace std;

// the moeoObjectiveVectorTraits : minimizing 2 objectives
class Sch1ObjectiveVectorTraits : public moeoObjectiveVectorTraits
  {
  public:
    static bool minimizing (int i)
    {
      return true;
    }
    static bool maximizing (int i)
    {
      return false;
    }
    static unsigned int nObjectives ()
    {
      return 2;
    }
  };


// objective vector of real values
typedef moeoRealObjectiveVector < Sch1ObjectiveVectorTraits > Sch1ObjectiveVector;


// multi-objective evolving object for the Sch1 problem
class Sch1 : public moeoRealVector < Sch1ObjectiveVector, double, double >
  {
  public:
    Sch1() : moeoRealVector < Sch1ObjectiveVector, double, double > (1)
    {}
  };


// evaluation of objective functions
class Sch1Eval : public moeoEvalFunc < Sch1 >
  {
  public:
    void operator () (Sch1 & _sch1)
    {
      if (_sch1.invalidObjectiveVector())
        {
          Sch1ObjectiveVector objVec;
          double x = _sch1[0];
          objVec[0] = x * x;
          objVec[1] = (x - 2.0) * (x - 2.0);
          _sch1.objectiveVector(objVec);
        }
    }
  };


// main
int main (int argc, char *argv[])
{
  // parameters
  unsigned int POP_SIZE = 20;
  unsigned int MAX_GEN = 100;
  double M_EPSILON = 0.01;
  double P_CROSS = 0.25;
  double P_MUT = 0.35;

  // objective functions evaluation
  Sch1Eval eval;

  // crossover and mutation
  eoQuadCloneOp < Sch1 > xover;
  eoUniformMutation < Sch1 > mutation (M_EPSILON);

  // generate initial population
  eoRealVectorBounds bounds (1, 0.0, 2.0);	// [0, 2]
  eoRealInitBounded < Sch1 > init (bounds);
  eoPop < Sch1 > pop (POP_SIZE, init);

  // build NSGA-II
  moeoNSGAII < Sch1 > nsgaII (MAX_GEN, eval, xover, P_CROSS, mutation, P_MUT);

  // run the algo
  nsgaII (pop);

  // extract first front of the final population using an moeoArchive (this is the output of nsgaII)
  moeoArchive < Sch1 > arch;
  arch.update (pop);

  // printing of the final archive
  cout << "Final Archive" << endl;
  arch.sortedPrintOn (cout);
  cout << endl;

  return EXIT_SUCCESS;
}
