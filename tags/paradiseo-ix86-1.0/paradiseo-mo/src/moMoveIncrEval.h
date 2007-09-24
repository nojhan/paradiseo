// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoMoveIncrEval.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moMoveIncrEval_h
#define __moMoveIncrEval_h

#include <eoFunctor.h>

//! (generally) Efficient evaluation function based a move and a solution.

/*!
  From a move and a solution, it computes
  a new fitness that could be associated to
  the solution if this one is updated.
*/
template < class M > class moMoveIncrEval:public eoBF < const M &, const typename
  M::EOType &,
  typename
  M::EOType::Fitness >
{

};

#endif
