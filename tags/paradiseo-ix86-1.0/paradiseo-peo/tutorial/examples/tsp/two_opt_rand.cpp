// "two_opt_rand.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <utils/eoRNG.h>

#include "two_opt_rand.h"
#include "node.h"  

void TwoOptRand :: operator () (TwoOpt & __move, const Route & __route) {

  __move.second = rng.random (numNodes);

  __move.first = rng.random (__move.second);
}
  

