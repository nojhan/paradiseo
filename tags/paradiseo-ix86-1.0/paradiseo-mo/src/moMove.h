// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moMove.h"

// (c) OPAC Team, LIFL, 2003-2006

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moMove_h
#define __moMove_h

#include <eoFunctor.h>

//! Definition of a move.

/*!
  A move transforms a solution to another close solution.
  It describes how a solution can be modified to another one.
*/
template < class EOT > class moMove:public eoUF < EOT &, void >
{

public:
  //! Alias for the type
  typedef EOT EOType;

};

#endif
