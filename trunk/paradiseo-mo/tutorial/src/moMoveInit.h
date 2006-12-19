// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moMoveInit.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moMoveInit_h
#define __moMoveInit_h

#include <eoFunctor.h>

//! Move (moMove) initializer
/*!
  Class which allows to initiase a move.
  Only a description... An object that herits from this class needs to be designed to be used.
 */
template < class M > class moMoveInit:public eoBF < M &, const typename
M::EOType &, void >
{

};

#endif
