// "route_eval.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __route_eval_h
#define __route_eval_h

#include <eoEvalFunc.h>

#include "route.h"

class RouteEval : public eoEvalFunc <Route> {
  
public :
    
  void operator () (Route & __route) ;  
} ;

#endif
