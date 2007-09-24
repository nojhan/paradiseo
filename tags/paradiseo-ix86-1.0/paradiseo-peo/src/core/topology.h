// "topology.h"

// (c) OPAC Team, LIFL, September 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __topology_h
#define __topology_h

#include <vector>

#include "cooperative.h"

class Topology {

public:

	virtual ~Topology ();

	void add (Cooperative & __mig); 

	virtual void setNeighbors (Cooperative * __mig,
				std :: vector <Cooperative *> & __from,
				std :: vector <Cooperative *> & __to) = 0;

protected:

	std :: vector <Cooperative *> mig ;  
};

#endif
