// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "city_swap.h"

// (c) OPAC Team, LIFL, 2002-2006

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef city_swap_h
#define city_swap_h

#include <eoOp.h>

#include "route.h"

/** Its swaps two vertices
    randomly choosen */
class CitySwap : public eoMonOp <Route> {
  
public :
  
  bool operator () (Route & __route) ;
    
} ;

#endif
