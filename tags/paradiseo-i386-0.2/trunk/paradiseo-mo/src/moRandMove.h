// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moRandMove.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moRandMove_h
#define __moRandMove_h

#include <eoFunctor.h>

//! Random move generator
/*!
  Only a description... An object that herits from this class needs to be designed in order to use a moSA. 
 */
template < class M > class moRandMove:public eoUF < M &, void >
{

};

#endif
