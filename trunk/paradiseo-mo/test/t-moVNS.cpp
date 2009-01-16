/*
* <t-moVNS.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2007-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Salma Mesmoudi (salma.mesmoudi@inria.fr), Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
* Jeremie Humeau (jeremie.humeau@inria.fr)
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
// t-moVNS.cpp
//-----------------------------------------------------------------------------

#include <eo>  // EO
#include <mo>  // MO
#include <cassert>

using std::cout;
using std::endl;

//-----------------------------------------------------------------------------

typedef EO<unsigned int> solution;
typedef eoScalarFitness< float, std::greater<float> > tspFitness ;
typedef eoVector <tspFitness, unsigned int> Route ;

int cpt=0;
int tableau[7]={10.0, 9.0, 8.0, 8.0, 7.0, 7.0, 7.0};

class testMove : public moMove <solution>
{
public :

  void operator () (solution & _solution)
  {
    solution sol=_solution;
  }
} ;


class solutionAlgo : public moAlgo <solution>
{
public :
  bool operator () (solution & _solution)
  {
    solution solution(_solution);
    return true;
  }
} ;

class Voisinage : public eoMonOp<Route>
{
public :
  bool operator () (Route & _solution)
  {
	_solution.invalidate();
    //_solution.fitness();

    return true;
  }
} ;

class Explorer : public moExpl<Route>
{
public:
	Explorer(eoMonOp<Route> & expl): moExpl<Route>( expl)
	  {

	  }
};

class solutionEval : public eoEvalFunc <Route>
{
public :
  void operator () (Route & _solution)
  {
	  _solution.fitness(tableau[cpt]);
	  cpt++;
  }
};

//-----------------------------------------------------------------------------

int
main()
{
  std::string test_result;
  int return_value;

  //solution solution;
   Route so ;

  Voisinage sol1;
  Voisinage sol2;
  Explorer explorer(sol1);
  explorer.add(sol2);
  solutionEval eval;

  moVNS<Route> vns(explorer, eval);

  cout << "[moVNS] ==> ";

  so.fitness(20.0);

  vns(so);

  assert(so.fitness()==7.0);

  cout << "OK" << endl;

  return EXIT_SUCCESS; //return_value;
}

//-----------------------------------------------------------------------------
