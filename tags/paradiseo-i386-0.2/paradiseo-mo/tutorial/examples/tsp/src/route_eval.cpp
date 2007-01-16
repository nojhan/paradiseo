// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "route_eval.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "route_eval.h"
#include "graph.h"

void RouteEval :: operator () (Route & __route) {
  
  float len = 0 ;
  
  for (unsigned i = 0 ; i < Graph :: size () ; i ++)
    len -= Graph :: distance (__route [i], __route [(i + 1) % Graph :: size ()]) ; 
  
  __route.fitness (len) ;
}
