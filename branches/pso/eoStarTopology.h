// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoStarTopology.h
// (c) OPAC 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef EOSTARTOPOLOGY_H_
#define EOSTARTOPOLOGY_H_

//-----------------------------------------------------------------------------
#include <eoTopology.h>
#include <eoNeighborhood.h>
//-----------------------------------------------------------------------------


/**
 * Topology dedicated to "globest best" strategy. 
 * All the particles of the swarm belong to the same and only one neighborhood.
 * The global best is stored as a protected member and updated by using the "update" method.
 */
template < class POT > class eoStarTopology:public eoTopology <POT>
{

public:

	/**
	 * The only Ctor. No parameter required.
	 */
    eoStarTopology (){}   


	/**
	 * Builds the only neighborhood that contains all the particles of the given population.
	 * Also initializes the global best particle with the best particle of the given population.
	 * @param _pop - The population used to build the only neighborhood.
	 */
    void setup(const eoPop<POT> & _pop)
    {  		
    	// put all the particles in the only neighborhood
    	for(unsigned i=0;i < _pop.size();i++)    	
    		neighborhood.put(i);  
    		
    	// set the initial global best as the best initial particle
    	neighborhood.best(_pop.best_element());
    }
    
    /*
     * Update the best fitness of the given particle if it's better.
     * Also replace the global best by the given particle if it's better.
     */
    void update(POT & _po,unsigned _indice)
    {
    	// update the best fitness of the particle
    	if(_po.fitness() > _po.best())
    	{
    		 _po.best(_po.fitness());
    	}	    	
    	// update the global best if the given particle is "better"
    	if(_po.fitness() > neighborhood.best().fitness())
    	{
    		neighborhood.best(_po);
    	}	
    }
    
    
    /**
     * Return the global best particle.
     */
    POT & best (unsigned  _indice) {return (neighborhood.best());}


    /**
     * Print the structure of the topology on the standard output.
     */
	void printOn()
	{
		std::cout << "{" ;
		for(unsigned i=0;i< neighborhood.size();i++)
			std::cout << neighborhood.get(i) << " ";	
		std::cout << "}" << std::endl;
	}


protected:
	eoNeighborhood<POT> neighborhood; // the only neighborhood

};

#endif /*EOSTARTOPOLOGY_H_ */








