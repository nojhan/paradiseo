// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moCoolSched.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moCoolSched_h
#define __moCoolSched_h

#include <eoFunctor.h>

//! This class gives the description of a cooling schedule.
/*!
  It is only a description... An object that herits from this class is needed to be used in a moSA.
  See moEasyCoolSched for example.
*/
class moCoolSched:public eoUF < double &, bool >
{

};

#endif
