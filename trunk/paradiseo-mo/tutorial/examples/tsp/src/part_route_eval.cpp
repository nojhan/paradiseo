// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "part_route_eval.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "part_route_eval.h"
#include "graph.h"

PartRouteEval :: PartRouteEval (float __from, float __to) : from (__from), to (__to) {}

void PartRouteEval :: operator () (Route & __route) 
{
  float len = 0 ;
  
  for (unsigned int i = (unsigned int) (__route.size () * from) ; i < (unsigned int ) (__route.size () * to) ; i ++)
    {
      len -= Graph :: distance (__route [i], __route [(i + 1) % Graph :: size ()]) ;
    }
  
  __route.fitness (len) ;
}
