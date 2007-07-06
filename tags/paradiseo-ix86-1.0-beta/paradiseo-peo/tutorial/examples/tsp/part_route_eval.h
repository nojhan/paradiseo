// "part_route_eval.h"

// (c) OPAC Team, LIFL, 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __part_route_eval_h
#define __part_route_eval_h

#include <eoEvalFunc.h>

#include "route.h"

/** Route Evaluator */
class PartRouteEval : public eoEvalFunc <Route> {
  
public :

  /** Constructor */
  PartRouteEval (float __from, float __to) ;
    
  void operator () (Route & __route) ;
  
private :

  float from, to ;

} ;


#endif
