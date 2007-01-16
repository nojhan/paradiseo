// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "part_two_opt_init.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef part_two_opt_init_h
#define part_two_opt_init_h

#include <eoMoveInit.h>

#include "two_opt.h"

/** It sets the first couple of edges */
class PartTwoOptInit : public eoMoveInit <TwoOpt> {
  
public :
  
  void operator () (TwoOpt & __move, const Route & __route) ;
  
} ;

#endif
