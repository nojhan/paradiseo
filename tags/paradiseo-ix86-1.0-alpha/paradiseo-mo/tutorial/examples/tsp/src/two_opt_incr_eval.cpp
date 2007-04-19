// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "TwoOptIncrEval.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "two_opt_incr_eval.h"
#include "graph.h"

float TwoOptIncrEval :: operator () (const TwoOpt & __move, const Route & __route) {
  
  // From
  unsigned v1 = __route [__move.first], v1_next = __route [__move.first + 1] ;
  
  // To
  unsigned v2 = __route [__move.second], v2_next = __route [__move.second + 1] ;
  
  return __route.fitness () - Graph :: distance (v1, v2) - Graph :: distance (v1_next, v2_next) + Graph :: distance (v1, v1_next) + Graph :: distance (v2, v2_next)  ;
}
