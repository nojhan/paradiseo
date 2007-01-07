// "route_init.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __route_init_h
#define __route_init_h

#include <eoInit.h>

#include "route.h"

class RouteInit : public eoInit <Route> {
  
public :
  
  void operator () (Route & __route);  
} ;

#endif
