// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt_init.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef two_opt_init_h
#define two_opt_init_h

#include <moMoveInit.h>

#include "two_opt.h"

/** It sets the first couple of edges */
class TwoOptInit : public moMoveInit <TwoOpt> 
{
  
public :
  
  void operator () (TwoOpt & __move, const Route & __route) ;
  
} ;

#endif
