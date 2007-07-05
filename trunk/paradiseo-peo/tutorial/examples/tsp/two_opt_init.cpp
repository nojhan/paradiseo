// "two_opt_init.cpp"

// (c) OPAC Team, LIFL, 2003

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "two_opt_init.h"

void TwoOptInit :: operator () (TwoOpt & __move, const Route & __route) {
  
  __move.first = __move.second = 0;
}
