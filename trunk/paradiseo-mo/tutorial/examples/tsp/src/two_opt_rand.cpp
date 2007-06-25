// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt_rand.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "two_opt_rand.h"
#include "graph.h"
#include <utils/eoRNG.h>

void TwoOptRand :: operator () (TwoOpt & __move) 
{
  __move.first = rng.random (Graph :: size () - 3) ;
  __move.second = __move.first + 2 + rng.random (Graph :: size () - __move.first - 3) ;
}
