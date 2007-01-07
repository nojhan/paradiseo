// "ring_topo.h"

// (c) OPAC Team, LIFL, September 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __ring_topo_h
#define __ring_topo_h

#include "topology.h"

class RingTopology : public Topology {
  
public :
   
  void setNeighbors (Cooperative * __mig,
		     std :: vector <Cooperative *> & __from,
		     std :: vector <Cooperative *> & __to);
  
};

#endif
