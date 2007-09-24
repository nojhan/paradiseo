// "city_swap.h"

// (c) OPAC Team, LIFL, 2002

/* 
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
