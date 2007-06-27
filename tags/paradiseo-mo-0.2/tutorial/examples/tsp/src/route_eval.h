// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "route_eval.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef route_eval_h
#define route_eval_h

#include <eoEvalFunc.h>

#include "route.h"

/** Route Evaluator */
class RouteEval : public eoEvalFunc <Route> 
{
  
public :
  
  void operator () (Route & __route) ;
  
} ;


#endif
