// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "order_xover.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef order_xover_h
#define order_xover_h

#include <eoOp.h>

#include "route.h"

/** Order Crossover */
class OrderXover : public eoQuadOp <Route> 
{
  
public :
  
  bool operator () (Route & __route1, Route & __route2) ;
  
private :
  
  void cross (const Route & __par1, const Route & __par2, Route & __child) ;
} ;

#endif
