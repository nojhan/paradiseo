// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoMoveExpl.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moMoveExpl_h
#define __moMoveExpl_h

#include <eoFunctor.h>

//! Description of a move (moMove) explorer
/*!
  Only a description...See moMoveLoopExpl.
 */
template < class M > class moMoveExpl:public eoBF < const typename
  M::EOType &,
  typename
M::EOType &, void >
{

};

#endif
