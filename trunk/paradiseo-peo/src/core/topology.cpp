// "topo.cpp"

// (c) OPAC Team, LIFL, September 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "topology.h"

Topology :: ~ Topology () {
  
  /* Nothing ! */
}

void Topology :: add (Cooperative & __mig) {
  
  mig.push_back (& __mig) ;
} 

