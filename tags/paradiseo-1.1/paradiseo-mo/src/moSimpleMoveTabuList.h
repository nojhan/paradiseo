/*
  <moSimpleMoveTabuList.h>
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

#ifndef _moSimpleMoveTabuList_h
#define _moSimpleMoveTabuList_h

#include <list>
#include <iterator>

#include <moTabuList.h>

//! Class describing a move tabu list with a limited memory.
template <class M>
class moSimpleMoveTabuList: public moTabuList < M >
{
 public:

  //! Alias for the type
  typedef typename M::EOType EOT;

  //! Alias for an iterator of a move list.
  typedef typename std::list<M>::iterator moveIterator;

  //! Constructor
  /*
    \param _size The maximum size of the move tabu list.
  */
 moSimpleMoveTabuList(unsigned int _memory_maximum_size): memory_maximum_size(_memory_maximum_size), memory_size(0)
    {}

  //! Function that indicates if, in a given state, the _move is tabu or not.
  /*!
    \param _move A given moMove.
    \param _solution A solution.
    \return true or false.
  */
  bool operator () (const M & _move, const EOT & _solution)
  {
    moveIterator it;
    //code only used to avoid warning because _solution is not used in this function.
    EOT solution=(EOT)_solution;
    
    it=tabuList.begin();
    // The code is !(*it)==_move instead of (*it)!=_move because people designing their specific move representation 
    // will write the "==" operator (I hope) but not necessary the "!=" operator.
    while ( it!=tabuList.end() && !((*it)==_move) )
      {
	it++;
      }

    return it!=tabuList.end();
  }

  void add(const M & _move, const EOT & _solution)
  {
    //code only used to avoid warning because _solution is not used in this function.
    const EOT solution(_solution);

    if (memory_size!=0)
      {
	// Useful in the case of a move has been kept thanks to the moAspirCrit.
	// In this case, the move can already be in the tabuList.
	removeMove(_move);
      }

    tabuList.push_back(_move);

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

  //! Procedure that removes a given move from the tabu list (if it is into, else do nothing).
  /*!
    \param _move A given moMove.
  */
  void removeMove(const M & _move)
  {
    moveIterator it;
    
    it=tabuList.begin();
    // The code is !(*it)==_move instead of (*it)!=_move because people designing their specific move representation 
    // will write the "==" operator (I hope) but not necessary the "!=" operator.
    while ( it!=tabuList.end() && (!((*it)==_move) ))
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

  //! The move tabu list.
  std::list<M> tabuList;
};

#endif
