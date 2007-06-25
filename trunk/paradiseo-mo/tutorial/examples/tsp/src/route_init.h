// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "route_init.h"

// (c) OPAC Team, LIFL, 2002-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef route_init_h
#define route_init_h

#include <eoInit.h>

#include "route.h"

class RouteInit : public eoInit <Route> 
{
  
public :
  
  void operator () (Route & __route) ;
  
} ;

#endif
