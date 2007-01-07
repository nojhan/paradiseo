// "route_eval.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "route_eval.h"

void RouteEval :: operator () (Route & __route) {
    
  __route.fitness (- (int) length (__route)); 
}
