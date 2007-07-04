// "two_opt_next.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "two_opt_next.h"
#include "node.h"

bool TwoOptNext :: operator () (TwoOpt & __move, const Route & __route) {

  if (__move.first == numNodes - 1 && __move.second == numNodes - 1)
    return false;
  
  else {
    
    __move.second ++;
    if (__move.second == numNodes) {
      
      __move.first ++;
      __move.second = __move.first;
    }
    return true ;
  }
}
