// "two_opt_next.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __two_opt_next_h
#define __two_opt_next_h

#include <moNextMove.h>

#include "two_opt.h"

class TwoOptNext : public moNextMove <TwoOpt> {

public :
  
  bool operator () (TwoOpt & __move, const Route & __route);
  
};

#endif
