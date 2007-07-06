// "merge_route_eval.cpp"

// (c) OPAC Team, LIFL, 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "merge_route_eval.h"

void MergeRouteEval :: operator () (Route & __route, const int & __part_fit) {

  int len = __route.fitness ();
  len += __part_fit;
  __route.fitness (len);
}
  
