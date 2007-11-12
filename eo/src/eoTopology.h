// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTopology.h
// (c) OPAC 2007
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: thomas.legrand@lifl.fr
             clive.canape@inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef EOTOPOLOGY_H_
#define EOTOPOLOGY_H_

//-----------------------------------------------------------------------------
#include <eoNeighborhood.h>
//-----------------------------------------------------------------------------


/**
 * Define the interface for a swarm optimization topology.
 */
template < class POT > class eoTopology:public eoPop < POT >
{
public:

    /**
     * Build the neighborhoods contained in the topology.
     */
    virtual void setup(const eoPop<POT> &)=0;

    /**
     * Update the neighborhood of the given particle and its indice in the population
     */
    virtual void updateNeighborhood(POT & ,unsigned)=0;


    /**
      * Update the neighborhood of the given particle thanks to a whole population (used for distributed or synchronous PSO)
      */
    virtual void updateNeighborhood(eoPop < POT > &_pop)
    {
        for (unsigned i = 0; i < _pop.size (); i++)
        {
            updateNeighborhood(_pop[i],i);
        }
    }

    /**
     * Build the neighborhoods contained in the topology.
     */
    virtual POT & best (unsigned ) = 0;
    
    /*
     * Return the global best
     */
    
    virtual POT & globalBest(const eoPop<POT>& _pop)
    {
    	POT globalBest,tmp;
    	unsigned indGlobalBest=0;
    	globalBest=best(0);
    	for(unsigned i=1;i<_pop.size();i++)
    	{
    		tmp=best(i);
    		if(globalBest.best() < tmp.best())
    		{
    			globalBest=tmp;
    			indGlobalBest=i;
    		}
    			
    	}
    	return best(indGlobalBest);
    }

    /**
     * Build the neighborhoods contained in the topology.
     * @param _pop - The population ton share between the neighborhood(s)
     */
    virtual void printOn(){}
};


#endif /*EOTOPOLOGY_H_ */








