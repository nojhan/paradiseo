// "TwoOptIncrEval.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __two_opt_incr_eval_h
#define __two_opt_incr_eval_h

#include <moMoveIncrEval.h>
#include "two_opt.h"

class TwoOptIncrEval : public moMoveIncrEval <TwoOpt> {

public :
  
  int operator () (const TwoOpt & __move, const Route & __route) ; 

} ;

#endif
