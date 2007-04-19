// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moSimpleSolutionTabuList.h"

// (c) OPAC Team, LIFL, 2003-2007

/* LICENCE TEXT 
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moSimpleSolutionTabuList_h
#define __moSimpleSolutionTabuList_h

#include <list>
#include <iterator>

#include "moTabuList.h"

//! Class describing a solution tabu list with limited length.
template <class M>
class moSimpleSolutionTabuList: public moTabuList < M >
{
  
public:

  //! Alias for the type
  typedef typename M::EOType EOT;
  
  //! Constructor
  /*!
    \param __size The maximum size of the solution tabu list.
   */
  moSimpleSolutionTabuList(unsigned __size): maxSize(__size)
  {
    currentSize=0;
  }
  
  //! Function that indicates if, in a given state, the _move is tabu or not.
  /*!
    \param __move A given moMove.
    \param __sol A solution.
    \return true or false.
  */
  bool operator () (const M & __move, const EOT & __sol) 
  {
    typename std::list<EOT>::iterator it;
  
    M _move=(M)__move;
    EOT _sol=(EOT) __sol;

    _move(_sol);
  
    it=tabuList.begin();
    while(it!=tabuList.end()&&(!((*it)==_sol)))
      {
	it++;
      }
    
    return it!=tabuList.end();
  }
  
  void
  add (const M & __move, const EOT & __sol)
  {
    M _move=(M)__move;
    EOT _sol=(EOT) _sol;
    
    _move(_sol);
    
    tabuList.push_back(_sol);
    
    if(currentSize==maxSize)
      {
	tabuList.erase(tabuList.begin());
      }
    else
      {
	currentSize++;
      }
  }

  void
  update ()
  {
    //nothing to do
  }

  void
  init ()
  {
    //nothing to do
  }

private:
  
  //! The maximum size of the tabu list.
  unsigned maxSize;

  //! The current size of the tabu list.
  unsigned currentSize;
  
  //! The solution tabu list.
  std::list<EOT> tabuList;
};

#endif
