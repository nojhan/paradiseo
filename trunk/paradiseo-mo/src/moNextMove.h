// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moNextMove.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moNextMove_h
#define __moNextMove_h

#include <eoFunctor.h>

//! Class which allows to generate a new move (moMove).
/*!
  Useful for the explorer (for moTS or moHC).
  Does nothing... An object that herits from this class needs to be designed for being used.
 */
template < class M > class moNextMove:public eoBF < M &, const typename
  M::EOType &,
  bool >
{

};

#endif
