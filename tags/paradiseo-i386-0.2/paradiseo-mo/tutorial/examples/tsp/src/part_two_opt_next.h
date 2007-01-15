// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "part_two_opt_next.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef part_two_opt_next_h
#define part_two_opt_next_h

#include <eoNextMove.h>
#include "two_opt.h"

/** It updates a couple of edges */
class PartTwoOptNext : public eoNextMove <TwoOpt> {

public :
  
  bool operator () (TwoOpt & __move, const Route & __route) ;
  
} ;

#endif
