// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoLinearTopology.h
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

#ifndef EOLINEARTOPOLOGY_H_
#define EOLINEARTOPOLOGY_H_

//-----------------------------------------------------------------------------
#include <eoPop.h>
#include <eoTopology.h>
#include <eoSocialNeighborhood.h>
//-----------------------------------------------------------------------------


/**
 *  One of the local best strategies for particle swarm optimization. Each particle has a fixed number of neighbours, ans
 *  the neighborhood is social.
 *  The topology is never modified during the flight.
 *
 *  @ingroup Selectors
 */
template < class POT > class eoLinearTopology:public eoTopology <
            POT >
{

public:

    /**
     * Build the topology made of _neighborhoodSize neighborhoods.
     * @param _neighborhoodSize - The size of each neighborhood.
     */
    eoLinearTopology (unsigned _neighborhoodSize):neighborhoodSize (_neighborhoodSize),isSetup(false){}


    /**
       * Build the neighborhoods contained in the topology.
       * @param _pop - The population used to build the neighborhoods.
       * If it remains several particle (because _pop.size()%neighborhoodSize !=0), there are inserted
       * in the last neighborhood. So it may be possible to have a bigger neighborhood.
       */
    void setup(const eoPop<POT> & _pop)
    {
        if (!isSetup)
        {
            // consitency check
            if (neighborhoodSize >= _pop.size()){
                std::string s;
                s.append (" Invalid neighborhood size in eoLinearTopology ");
                throw std::runtime_error (s);
            }

            unsigned howManyNeighborhood=_pop.size()/ neighborhoodSize;

            // build all the neighborhoods
            for (unsigned i=0;i< howManyNeighborhood;i++)
            {
                eoSocialNeighborhood<POT> currentNghd;

                currentNghd.best(_pop[i*neighborhoodSize]);
                for (unsigned k=i*neighborhoodSize;k < neighborhoodSize*(i+1);k++)
                {
                    currentNghd.put(k);
                    if (_pop[k].fitness() > currentNghd.best().fitness())
                        currentNghd.best(_pop[k]);
                }
                neighborhoods.push_back(currentNghd);
            }

            // assign the last neighborhood to the remaining particles
            if (_pop.size()%neighborhoodSize !=0)
            {
                for (unsigned z=_pop.size()-1;z >= (_pop.size()-_pop.size()%neighborhoodSize);z--){
                    neighborhoods.back().put(z);

                    if (_pop[z].fitness() > neighborhoods.back().best().fitness())
                        neighborhoods.back().best(_pop[z]);
                }
            }

            isSetup=true;
        }
        else
        {
            // Should activate this part ?
            /*
               std::string s;
               s.append (" Linear topology already setup in eoLinearTopology");
               throw std::runtime_error (s);
               */
        }

    }


    /**
     * Retrieve the neighboorhood of a particle.
     * @return _indice - The particle indice (in the population)
     */
    unsigned retrieveNeighborhoodByIndice(unsigned _indice)
    {
        unsigned i=0;
        for (i=0;i< neighborhoods.size();i++)
        {
            if (neighborhoods[i].contains(_indice))
            {
                return i;
            }
        }
        return i;
    }

   /**
    * Update the neighborhood: update the particle's best fitness and the best particle
    * of the corresponding neighborhood.
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

        // update the best in its neighborhood
        unsigned theGoodNhbd= retrieveNeighborhoodByIndice(_indice);
        if (_po.fitness() > neighborhoods[theGoodNhbd].best().fitness())
        {
            neighborhoods[theGoodNhbd].best(_po);
        }
    }

    /**
    * Return the best informative of a particle. Could be itself.
    * @param _indice - The indice of a particle in the population
    * @return POT & - The best particle in the neighborhood of the particle whose indice is _indice
    */
    POT & best (unsigned  _indice)
    {
        unsigned theGoodNhbd= retrieveNeighborhoodByIndice(_indice);

        return (neighborhoods[theGoodNhbd].best());
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

    /**
     * Print the structure of the topology on the standrad output.
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


protected:
        std::vector<eoSocialNeighborhood<POT> >  neighborhoods;
    unsigned neighborhoodSize; // the size of each neighborhood

    bool isSetup;

};

#endif /*EOLINEARTOPOLOGY_H_ */
