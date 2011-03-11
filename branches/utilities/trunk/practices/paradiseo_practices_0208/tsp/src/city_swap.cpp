// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "city_swap.cpp"

// (c) OPAC Team, LIFL, 2002-2006

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <utils/eoRNG.h>

#include "city_swap.h"

bool CitySwap :: operator () (Route & __route) {
  
  std :: swap (__route [rng.random (__route.size ())],
	       __route [rng.random (__route.size ())]) ;
    
  __route.invalidate () ;
  
  return true ;
}
