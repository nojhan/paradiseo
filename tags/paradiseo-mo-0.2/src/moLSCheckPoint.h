// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moLSCheckPoint.h"

// (c) OPAC Team, LIFL, 2003

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moSolUpdater_h
#define __moSolUpdater_h

#include <eoFunctor.h>

//! Class which allows a checkpointing system.
/*!
  Thanks to this class, at each iteration, additionnal function can be used (and not only one).
*/
template < class M > class moLSCheckPoint:public eoBF < const M &, const typename
  M::EOType &, void >
{

public:
  //! Function which launches the checkpointing
  /*!
     Each saved function is used on the current move and the current solution.

     \param __move a move.
     \param __sol a solution.
   */
  void
  operator   () (const M & __move, const typename M::EOType & __sol)
  {

    for (unsigned int i = 0; i < func.size (); i++)
      {
	func[i]->operator   ()(__move, __sol);
      }
  }

  //! Procedure which add a new function to the function vector
  /*!
     The new function is added at the end of the vector.
     \param __f a new function to add.
   */
  void
  add (eoBF < const M &, const typename M::EOType &, void >&__f)
  {

    func.push_back (&__f);
  }

private:

  //! vector of function
  std::vector < eoBF < const
    M &, const
    typename
  M::EOType &, void >*>
    func;

};

#endif
