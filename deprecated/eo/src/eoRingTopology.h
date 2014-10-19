// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRingTopology.h
// (c) INRIA Futurs DOLPHIN 2007
/*
    Clive Canape
        Thomas Legrand

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

    Contact: clive.canape@inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef EORINGTOPOLOGY_H_
#define EORINGTOPOLOGY_H_

//-----------------------------------------------------------------------------
#include <eoTopology.h>
#include <eoSocialNeighborhood.h>
//-----------------------------------------------------------------------------


/**
 * Static ring topology for particle swarm optimization.
 * The neighborhoods are built using a ring based on each particle's indice and
 * do not change for all the time steps. Only the best particle in each neighborhood is
 * potentially updated thanks to the "updateNeighborhood" method.
 *
 * @ingroup Selectors
 */
template < class POT > class eoRingTopology:public eoTopology <POT>
{

public:

    /**
     * The only Ctor.
     * @param _neighborhoodSize - The size of each neighborhood.
     */
    eoRingTopology (unsigned _neighborhoodSize):neighborhoodSize (_neighborhoodSize),isSetup(false){}


    /**
     * Builds the neighborhoods using a ring strategy based on the particle indices.
     * Also initializes the best particle of each neighborhood.
     * @param _pop - The population used to build the only neighborhood.
     * @return
     */
    void setup(const eoPop<POT> & _pop)
    {
        if (!isSetup){

            // put in the neighborhood
            int k = neighborhoodSize/2;
            for (unsigned i=0;i < _pop.size();i++)
            {
                eoSocialNeighborhood<POT> currentNghd;
                currentNghd.best(_pop[i]);
                for (unsigned j=0; j < neighborhoodSize; j++)
                {
                        currentNghd.put((_pop.size()+i-k+j)%_pop.size());
                        if(_pop[(_pop.size()+i-k+j)%_pop.size()].fitness() > currentNghd.best().fitness())
                        currentNghd.best(_pop[(_pop.size()+i-k+j)%_pop.size()]);
                }
                neighborhoods.push_back(currentNghd);
            }
            isSetup=true;
        }
        else
        {
            // Should activate this part ?
            /*
               std::string s;
               s.append (" Linear topology already setup in eoRingTopology");
               throw std::runtime_error (s);
               */
        }
    }

    /**
     * Retrieves the neighboorhood of a particle.
     * @return _indice - The particle indice (in the population)
     */
    unsigned retrieveNeighborhoodByIndice(unsigned _indice)
    {
        return _indice;
    }


    /**
     * Updates the best fitness of the given particle and
     * potentially replaces the local best the given particle it's better.
     * @param _po - The particle to update
     * @param _indice - The indice of the given particle in the population
     */
    void updateNeighborhood(POT & _po,unsigned _indice)
    {
        //this->printOn();exit(0);
        // update the best fitness of the particle
        if (_po.fitness() > _po.best())
        {
            _po.best(_po.fitness());
            for(unsigned i=0;i<_po.size();i++)
                        _po.bestPositions[i]=_po[i];
        }
        // update the global best if the given particle is "better"
        for (unsigned i=-neighborhoodSize+1; i < neighborhoodSize; i++)
                {
                        unsigned indi = (_po.size()+_indice+i)%_po.size();
                        if (_po.fitness() > neighborhoods[indi].best().fitness())
                                neighborhoods[indi].best(_po);
                }
     }


    /**
     * Returns the best particle belonging to the neighborhood of the given particle.
     * @param _indice - The indice of a particle in the population
     * @return POT & - The best particle in the neighborhood of the particle whose indice is _indice
     */
    POT & best (unsigned  _indice)
    {
        unsigned theGoodNhbd= retrieveNeighborhoodByIndice(_indice);

        return (neighborhoods[theGoodNhbd].best());
    }


    /**
     * Print the structure of the topology on the standard output.
     */
    void printOn()
    {
        for (unsigned i=0;i< neighborhoods.size();i++)
        {
            std::cout << "{ " ;
            for (unsigned j=0;j< neighborhoods[i].size();j++)
            {
                std::cout << neighborhoods[i].get(j) << " ";
            }
            std::cout << "}" << std::endl;
        }
    }

    /*
         * Return the global best of the topology
         */
        virtual POT & globalBest()
    {
        POT gBest,tmp;
        unsigned indGlobalBest=0;
        if(neighborhoods.size()==1)
                return neighborhoods[0].best();

        gBest=neighborhoods[0].best();
        for(unsigned i=1;i<neighborhoods.size();i++)
        {
                tmp=neighborhoods[i].best();
                if(gBest.best() < tmp.best())
                {
                        gBest=tmp;
                        indGlobalBest=i;
                }

        }
        return neighborhoods[indGlobalBest].best();
    }


protected:
    std::vector<eoSocialNeighborhood<POT> >  neighborhoods;
    unsigned neighborhoodSize;
    bool isSetup;
};
/** @example t-eoRingTopology.cpp
 */

#endif /*EORINGTOPOLOGY_H_*/
