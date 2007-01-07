// "two_opt_rand.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __two_opt_rand_h
#define __two_opt_rand_h

#include <eoMoveRand.h>

#include "two_opt.h"

class TwoOptRand : public eoMoveRand <TwoOpt> {
  
public :
  
  void operator () (TwoOpt & __move, const Route & __route) ;
  
} ;

#endif
