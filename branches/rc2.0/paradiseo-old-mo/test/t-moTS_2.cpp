/*
* <t-moTS_2.cpp>
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
// t-moTS_2.cpp
//-----------------------------------------------------------------------------

#include <eo>  // EO
#include <oldmo>  // MO

using std::cout;
using std::endl;

//-----------------------------------------------------------------------------

typedef EO<unsigned int> solution;

class testMove : public moMove <solution>
{
public :
  void operator () (solution & _solution)
  {
    solution sol=_solution;
  }
} ;

class testMoveExplorer : public moMoveExpl<testMove>
{
public :
  void operator () (const solution & _solution_1, solution & _solution_2)
  {
    const solution solution(_solution_1);
    _solution_2.fitness(2);
  }
};

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

class solutionEval : public eoEvalFunc <solution>
{
public :
  void operator () (solution & _solution)
  {
    _solution.fitness(0);
  }
} ;


//-----------------------------------------------------------------------------

int
main()
{
  std::string test_result;
  int return_value;

  solution solution;

  testMoveExplorer explorer;
  solutionContinue continu;
  solutionEval eval;
  testMove move;

  moTS<testMove> ts(explorer, continu, eval);

  cout << "[ moTS_2                       ] ==> ";

  ts(solution);

  move(solution);

  test_result=((solution.fitness()!=2)?"KO":"OK");
  return_value=((test_result.compare("KO")==0)?EXIT_FAILURE:EXIT_SUCCESS);

  cout << test_result << endl;
  return return_value;
}

//-----------------------------------------------------------------------------
