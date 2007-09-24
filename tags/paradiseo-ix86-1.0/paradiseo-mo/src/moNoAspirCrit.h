// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoNoAspirCrit.h"

// (c) OPAC Team, LIFL, 2003-2006

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moNoAspirCrit_h
#define __moNoAspirCrit_h

#include "moAspirCrit.h"

//! One of the possible aspiration criterion (moAspirCrit)
/*!
  The simplest : never satisfied.
 */
template < class M > class moNoAspirCrit:public moAspirCrit < M >
{

  //! Function which describes the aspiration criterion behaviour
  /*!
     Does nothing.

     \param __move a move.
     \param __sol a fitness.
     \return FALSE.
   */
  bool operator   () (const M & __move,
		      const typename M::EOType::Fitness & __sol)
  {

    return false;
  }

  //! Procedure which initialises all that needs a moNoAspirCrit
  /*!
     Nothing...
   */
  void init ()
  {
  }
};

#endif
