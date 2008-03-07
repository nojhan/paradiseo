/*
* <t-moSA.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2008
*
* Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
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
// t-moSA.cpp
//-----------------------------------------------------------------------------

#include <eo>  // EO
#include <mo>  // MO

using std::cout;
using std::endl;

//-----------------------------------------------------------------------------

typedef EO<int> solution;

class testMove : public moMove <solution>
{
public :
  void operator () (solution & _solution)
  {
    solution sol=_solution;
  }
} ;

class testRandMove : public moRandMove<testMove>
{
public :
  void operator () (testMove & _move)
  {
    _move=_move;
  }
};

class testMoveIncrEval : public moMoveIncrEval <testMove>
{
public :
  int operator () (const testMove & _move, const solution & _solution)
  {
    const testMove move(_move);
    const solution solution(_solution);

    return 2;
  }
} ;

class solutionContinue : public moSolContinue<solution>
{
public :
  bool operator () (const solution & _solution)
  {
    const solution sol(_solution);
    
    return false;
  }

  void init()
  {}
} ;

class testCooling : public moCoolingSchedule
{
public :
  bool operator () (double & _temperature)
  {
    double temperature;
    temperature=_temperature;

    return false;
  }
};

class solutionEval : public eoEvalFunc <solution>
{
public :
  void operator () (solution & _solution)
  {
    _solution.fitness(2);
  }
} ;


//-----------------------------------------------------------------------------

int
main()
{
  cout << "[ moSA                         ] ==> ";
  
  solution sol;

  sol.fitness(0);

  testRandMove rand;
  testMoveIncrEval incrEval;
  solutionContinue continu;
  testCooling cooling;
  solutionEval eval;

  moSA<testMove> sa(rand, incrEval, continu, 10.0, cooling, eval);

  sa(sol);

  if(sol.fitness()!=2)
    {
      cout << "KO" << endl;
      return EXIT_FAILURE;
    }
  
  cout << "OK" << endl;
  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
