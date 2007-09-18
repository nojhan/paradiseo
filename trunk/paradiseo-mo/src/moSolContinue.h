// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moSolContinue.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moSolContinue_h
#define __moSolContinue_h

#include <eoFunctor.h>

//! Class that describes a stop criterion for a solution-based heuristic

/*! 
  It allows to add an initialisation procedure to an object that is a unary function (eoUF).
*/
template < class EOT > class moSolContinue:public eoUF < const EOT &, bool >
{

public:
  //! Procedure which initialises all that the stop criterion needs
  /*!
     Generally, it allocates some data structures or initialises some counters.
   */
  virtual void init () = 0;
};

#endif
