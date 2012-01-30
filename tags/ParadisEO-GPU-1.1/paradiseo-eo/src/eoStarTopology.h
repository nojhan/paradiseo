// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoStarTopology.h
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

#ifndef EOSTARTOPOLOGY_H_
#define EOSTARTOPOLOGY_H_

//-----------------------------------------------------------------------------
#include <eoTopology.h>
#include <eoSocialNeighborhood.h>
//-----------------------------------------------------------------------------


/**
 * Topology dedicated to "globest best" strategy for particle swarm optimization.
 * All the particles of the swarm belong to the same and only social neighborhood.
 * The global best is stored and updated using the eoSocialNeighborhood.
*
*   @ingroup Selectors
 */
template < class POT > class eoStarTopology:public eoTopology <POT>
{

public:

    /**
     * The only Ctor. No parameter required.
     */
    eoStarTopology ():isSetup(false){}


    /**
     * Builds the only neighborhood that contains all the particles of the given population.
     * Also initializes the global best particle with the best particle of the given population.
     * @param _pop - The population used to build the only neighborhood.
     * @return
     */
    void setup(const eoPop<POT> & _pop)
    {
        if (!isSetup){

            // put all the particles in the only neighborhood
            for (unsigned i=0;i < _pop.size();i++)
                neighborhood.put(i);

            // set the initial global best as the best initial particle
            neighborhood.best(_pop.best_element());

            isSetup=true;
        }
        else
        {
            // Should activate this part ?
            /*
               std::string s;
               s.append (" Linear topology already setup in eoStarTopology");
               throw std::runtime_error (s);
               */
        }
    }

   /*
     * Update the best fitness of the given particle if it's better.
     * Also replace the global best by the given particle if it's better.
     * @param _po - The particle to update
     * @param _indice - The indice of the given particle in the population
     */
    void updateNeighborhood(POT & _po,unsigned _indice)
    {
        // update the best fitness of the particle
        if (_po.fitness() > _po.best())
        {
          _po.best(_po.fitness());
          for(unsigned i=0;i<_po.size();i++)
            _po.bestPositions[i]=_po[i];
        }
        // update the global best if the given particle is "better"
        if (_po.fitness() > neighborhood.best().fitness())
        {
            neighborhood.best(_po);
        }
    }


    /**
     * Return the global best particle.
     * @param _indice - The indice of a particle in the population
     * @return POT & - The best particle in the neighborhood of the particle whose indice is _indice
     */
    POT & best (unsigned  _indice) {return (neighborhood.best());}

    /*
         * Return the global best of the topology
         */

         virtual POT & globalBest(const eoPop<POT>& _pop)
    {
        return neighborhood.best();
    }


   /**
     * Print the structure of the topology on the standard output.
     */
    void printOn()
    {
        std::cout << "{" ;
        for (unsigned i=0;i< neighborhood.size();i++)
            std::cout << neighborhood.get(i) << " ";
        std::cout << "}" << std::endl;
    }



protected:
    eoSocialNeighborhood<POT> neighborhood; // the only neighborhood
    bool isSetup;
};

#endif /*EOSTARTOPOLOGY_H_ */
