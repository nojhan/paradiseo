// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moSimpleMoveTabuList.h"

// (c) OPAC Team, LIFL, 2003-2007

/* LICENCE TEXT 
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moSimpleMoveTabuList_h
#define __moSimpleMoveTabuList_h

#include <list>
#include <iterator>

#include "moTabuList.h"

//! Class describing a move tabu list with a limited memory.
template <class M>
class moSimpleMoveTabuList: public moTabuList < M >
{
  
public:
  
  //! Alias for the type
  typedef typename M::EOType EOT;

  //! Constructor
  /*
    \param __size The maximum size of the move tabu list.
   */
  moSimpleMoveTabuList(unsigned __size): maxSize(__size)
  {
    currentSize=0;
  }

   //! Function that indicates if, in a given state, the _move is tabu or not.
  /*!
    \param __move A given moMove.
    \param __sol A solution.
    \return true or false.
  */
  bool
  operator () (const M & __move, const EOT & __sol) 
  {
    typename std::list<M>::iterator it;
    
    it=tabuList.begin();
    while(it!=tabuList.end()&&(!((*it)==__move)))
      {
	it++;
      }
    
    return it!=tabuList.end();
  }
  
  void
  add (const M & __move, const EOT & __sol)
  {
    tabuList.push_back(__move);
    
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
  
  //! The move tabu list.
  std::list<M> tabuList;
};

#endif
