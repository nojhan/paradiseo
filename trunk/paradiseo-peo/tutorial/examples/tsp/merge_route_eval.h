// "merge_route_eval.h"

// (c) OPAC Team, LIFL, 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __merge_route_eval_h
#define __merge_route_eval_h

#include <peoAggEvalFunc.h>

#include "route.h"

class MergeRouteEval : public peoAggEvalFunc <Route> {
  
public :

  void operator () (Route & __route, const int & __part_fit) ;
  
};

#endif
