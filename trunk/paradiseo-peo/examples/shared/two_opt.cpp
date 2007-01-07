// "two_opt.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "two_opt.h"

void TwoOpt :: operator () (Route & __route) {
  
  unsigned i = 0; 

  while ((2 * i) < (second - first)) {
    
    std :: swap (__route [first + i], __route [second - i]);
    i ++;
  }
}
