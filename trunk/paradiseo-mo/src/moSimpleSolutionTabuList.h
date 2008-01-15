/*
  <moSimpleSolutionTabuList.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
  (C) OPAC Team, LIFL, 2002-2008
 
  Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
 
  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
 
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.
 
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.
 
  ParadisEO WebSite : http://paradiseo.gforge.inria.fr
  Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef _moSimpleSolutionTabuList_h
#define _moSimpleSolutionTabuList_h

#include <list>
#include <iterator>

#include <moTabuList.h>

//! Class describing a solution tabu list with limited length.
template <class M>
class moSimpleSolutionTabuList: public moTabuList < M >
{
 public:

  //! Alias for the type
  typedef typename M::EOType EOT;

  //! Alias for an iterator of a solution list.
  typedef typename std::list<EOT>::iterator solutionIterator;
  
  //! Constructor
  /*!
    \param _memory_maximum_size The maximum size of the solution tabu list.
  */
 moSimpleSolutionTabuList(unsigned int _memory_maximum_size): memory_maximum_size(_memory_maximum_size), memory_size(0)
    {}

  //! Function that indicates if, in a given state, the _move is tabu or not.
  /*!
    \param _move A given moMove.
    \param _solution A solution.
    \return true or false.
  */
  bool operator () (const M & _move, const EOT & _solution)
  {
    solutionIterator it;

    M move=(M)_move;
    EOT solution=(EOT) _solution;

    move(solution);

    it=tabuList.begin();
    // The code is !(*it)==_solution instead of (*it)!=_solution because people designing their specific solution representation 
    // will write the "==" operator (I hope) but not necessary the "!=" operator.
    while (it!=tabuList.end()&&(!((*it)==solution)))
      {
	it++;
      }

    return it!=tabuList.end();
  }

  void add (const M & _move, const EOT & _solution)
  {
    M move=(M)_move;
    EOT solution=(EOT) _solution;

    _move(_solution);

    if (memory_size!=0)
      {
	// Useful in the case of a solution has been kept thanks to the moAspirCrit.
	// In this case, the solution can already be in the tabuList.
	removeSolution(_solution);
      }

    tabuList.push_back(_solution);

    if (memory_size == memory_maximum_size)
      {
	tabuList.erase(tabuList.begin());
      }
    else
      {
	memory_size++;
      }
  }

  void update ()
  {
    //nothing to do
  }

  void init ()
  {
    //nothing to do
  }

 private:

  //! Procedure that removes a given solution from the tabu list (if it is into, else does nothing).
  /*!
    \param _solution A given solution.
  */
  void removeSolution(const EOT & _solution)
  {
    solutionIterator it;

    it=tabuList.begin();
    // The code is !(*it)==_solution instead of (*it)!=_solution because people designing their specific solution representation 
    // will write the "==" operator (I hope) but not necessary the "!=" operator.
    while ( it!=tabuList.end() && !((*it)==_solution) )
      {
	it++;
      }

    if (it!=tabuList.end())
      {
	tabuList.erase(it);
      }
  }

  //! The maximum size of the tabu list.
  unsigned int memory_maximum_size;

  //! The current size of the tabu list.
  unsigned int memory_size;

  //! The solution tabu list.
  std::list<EOT> tabuList;
};

#endif
