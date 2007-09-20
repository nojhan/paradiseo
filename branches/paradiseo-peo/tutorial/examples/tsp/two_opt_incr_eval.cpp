// "TwoOptIncrEval.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "two_opt_incr_eval.h"
#include "node.h"

int TwoOptIncrEval :: operator () (const TwoOpt & __move, const Route & __route) {
  
  /* From */
  Node v1 = __route [__move.first], v1_left = __route [(__move.first - 1 + numNodes) % numNodes];
  
  /* To */
  Node v2 = __route [__move.second], v2_right = __route [(__move.second + 1) % numNodes];
 
  if (v1 == v2 || v2_right == v1)
    return __route.fitness ();
  else 
    return __route.fitness () - distance (v1_left, v2) - distance (v1, v2_right) + distance (v1_left, v1) + distance (v2, v2_right);
}
