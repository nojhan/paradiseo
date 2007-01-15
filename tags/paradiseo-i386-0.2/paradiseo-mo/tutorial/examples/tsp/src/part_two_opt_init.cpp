// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "part_two_opt_init.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <utils/eoRNG.h>

#include "part_two_opt_init.h"

void PartTwoOptInit :: operator () (TwoOpt & __move, const Route & __route) {
  
  __move.first = rng.random (__route.size () - 6) ;
  __move.second = __move.first + 2 ;
}
