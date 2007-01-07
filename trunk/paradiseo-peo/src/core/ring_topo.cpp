// "ring_topo.cpp"

// (c) OPAC Team, LIFL, September 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "ring_topo.h"

void RingTopology :: setNeighbors (Cooperative * __mig,
				   std :: vector <Cooperative *> & __from,
				   std :: vector <Cooperative *> & __to) {
  __from.clear () ;
  __to.clear () ;

    int len = mig.size () ;
    
    for (int i = 0 ; i < len ; i ++)      
      if (mig [i] == __mig) {	
	__from.push_back (mig [(i - 1 + len) % len]) ;
	__to.push_back (mig [(i + 1) % len]) ;	
	break;
      }
}
