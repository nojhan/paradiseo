// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoSocialNeighborhood.h
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
 */
//-----------------------------------------------------------------------------

#ifndef EOSOCIALNEIGHBORHOOD_H_
#define EOSOCIALNEIGHBORHOOD_H_

//-----------------------------------------------------------------------------
#include <eoNeighborhood.h>
//-----------------------------------------------------------------------------

/**
 *  Derivated from eoNeighborhood. Just takes relationships into account.
 * The neighborhood is defined as a list of indices corresponding to particles.
 * Also contains ONE particle considered as the best of the neighborhood.
 *
 * @ingroup Selectors
 */
template < class POT > class eoSocialNeighborhood : public eoNeighborhood<POT>
{
public:

    eoSocialNeighborhood(){}

    /**
     * Put a particle (identified by its indice in its population) in the neighborhood.
     * @param _oneIndice - The indice of the particle in its population.
     */
    void put(unsigned _oneIndice)
    {
        indicesList.push_back(_oneIndice);
    }

    /**
     * Return true if the neighborhood contains the indice (= that means "contains the
     * particle whose indice is _oneIndice")
     * @param _oneIndice - The indice of the particle in its population.
     */
    bool contains(unsigned _oneIndice)
    {
        for (unsigned i=0;i< indicesList.size();i++)
        {
            if (indicesList[i]==_oneIndice)
                return true;
        }
        return false;
    }

    /**
     * Return the list of particle indices as a vector.
     */
    std::vector<unsigned> getInformatives()
    {
        return indicesList;
    }

    /**
     * Return the size of the neighborhood.
     */
    unsigned size()
    {
        return indicesList.size();

    }

    /**
     * Return the "_index-th" particle of the neighborhood.
     * Throw an exception if its not contained in the neighborhood.
     */
    unsigned get(unsigned _index)
    {
        if (_index < size())
            return indicesList[_index];
        else{
            std::string s;
            s.append (" Invalid indice in eoSocialNeighborhood ");
            throw std::runtime_error (s);
        }
    }

    /**
     * Return the best particle of the neighborhood.
     * The topology is expected to get it.
     */
    POT & best()
    {
        return lBest;
    }

    /**
     * Set the best particle of the neighborhood.
     * The topology is expected to set it.
     */
    void best(POT _particle)
    {
        lBest=_particle;
    }

protected:
    std::vector<unsigned> indicesList; // The list of particles as a vector of indices
    POT lBest; // the best particle of the neighborhood
};


#endif /* EOSOCIALNEIGHBORHOOD_H_ */
