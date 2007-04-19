// "part_route_eval.cpp"

// (c) OPAC Team, LIFL, 2003

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "part_route_eval.h"
#include "node.h"

PartRouteEval :: PartRouteEval (float __from,
				float __to
				) : from (__from),
				    to (__to) {
  
}

void PartRouteEval :: operator () (Route & __route) {
  
  
  unsigned len = 0 ;
  
  for (unsigned i = (unsigned) (__route.size () * from) ;
       i < (unsigned) (__route.size () * to) ;
       i ++)
    len += distance (__route [i], __route [(i + 1) % numNodes]) ;
  
  __route.fitness (- (int) len) ;
}
