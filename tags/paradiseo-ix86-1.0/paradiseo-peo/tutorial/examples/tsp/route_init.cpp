// "route_init.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <utils/eoRNG.h>

#include "route_init.h"
#include "node.h"

void RouteInit :: operator () (Route & __route) {

  __route.clear ();
  
  for (unsigned i = 0 ; i < numNodes ; i ++)
    __route.push_back (i);
  
  for (unsigned i = 0 ; i < numNodes ; i ++)    
    std :: swap (__route [i], __route [rng.random (numNodes)]);
}
