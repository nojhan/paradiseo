// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "part_route_eval.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT 
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef part_route_eval_h
#define part_route_eval_h

#include <eoEvalFunc.h>

#include "route.h"

/** Route Evaluator */
class PartRouteEval : public eoEvalFunc <Route> 
{
  
public :
  
  /** Constructor */
  PartRouteEval (float __from, float __to) ;
  
  void operator () (Route & __route) ;
  
private :

  float from, to ;
  
} ;


#endif
