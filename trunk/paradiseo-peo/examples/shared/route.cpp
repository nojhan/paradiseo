// "route.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "route.h"

unsigned length (const Route & __route) {

  unsigned len = 0 ;
  
  for (unsigned i = 0; i < numNodes; i ++)
    len += distance (__route [i], __route [(i + 1) % numNodes]) ; 
  
  return len;
}


