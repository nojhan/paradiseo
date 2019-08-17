/*
 <moBitsNeighborhood.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010
 
 Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau
 
 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".
 
 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited liability.
 
 In this respect, the user's attention is drawn to the risks associated
 with loading,  using,  modifying and/or developing or reproducing the
 software by the user in light of its specific status of free software,
 that may mean  that it is complicated to manipulate,  and  that  also
 therefore means  that it is reserved for developers  and  experienced
 professionals having in-depth computer knowledge. Users are therefore
 encouraged to load and test the software's suitability as regards their
 requirements in conditions enabling the security of their systems and/or
 data to be ensured and,  more generally, to use and operate it in the
 same conditions as regards security.
 The fact that you are presently reading this means that you have had
 knowledge of the CeCILL license and that you accept its terms.
 
 ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 Contact: paradiseo-help@lists.gforge.inria.fr
 */

#ifndef _moBitsNeighborhood_h
#define _moBitsNeighborhood_h

#include "../../neighborhood/moNeighborhood.h"
#include <paradiseo/eo/utils/eoRNG.h>
#include <vector>

/**
 * A neighborhood for bit string solutions
 * where several bits could be flipped
 * in a given Hamming distance
 */
template< class Neighbor >
class moBitsNeighborhood : public moNeighborhood<Neighbor>
{
public:
    
    /**
     * Define type of a solution corresponding to Neighbor
     */
    typedef typename Neighbor::EOT EOT;
    
    /**
     * Constructor
     * @param _length bit string length
     * @param _nBits maximum number of bits to flip (radius of the neighborhood)
     * @param _exactDistance when true, only neighbor with exactly k bits flip are considered, other neighbor <= Hamming distance k
     */
    moBitsNeighborhood(unsigned _length, unsigned _nBits, bool _exactDistance = false): moNeighborhood<Neighbor>(), length(_length), nBits(_nBits) {
        // neighborhood size :
        // for distance == nBits : length \choose nBits = length! / ( (length - nBits)! * nBits!)
        // for distance <= nBits : sum of previous distances
        if (_exactDistance) {
            neighborhoodSize = numberOfNeighbors(nBits);
        } else {
            neighborhoodSize = 0;
            for(int d = 1; d <= nBits; d++)
                neighborhoodSize += numberOfNeighbors(d);
        }
        
    }
    
    /**
     * Number fo neighbors at Hamming distance d
     *
     * @param d Hamming distance
     */
    unsigned int numberOfNeighbors(unsigned d) {
        unsigned int fact_nBits = 1;
        
        for(unsigned k = 1; k <= d; k++)
            fact_nBits *= k;
        
        unsigned int fact_length = 1;
        
        for(unsigned k = length; k > length - d; k--)
            fact_length *= k;
        
        return fact_length / fact_nBits;
    }
    
    /**
     * Test if it exist a neighbor
     * @param _solution the solution to explore
     * @return true if the neighborhood was not empty (bit string larger than 0)
     */
    virtual bool hasNeighbor(EOT& _solution) {
        return _solution.size() > 0;
    }
    
    /**
     * Return the class Name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moBitsNeighborhood";
    }
    
    /**
     * The neighborhood is random here
     * @return true, since the neighborhood is random
     */
    bool isRandom() {
        return true;
    }
    
protected:
    // length of the bit strings
    unsigned int length;
    
    // radius of the neighborhood
    unsigned int nBits;
    
    // size of the neighborhood
    unsigned int neighborhoodSize;
    
};

#endif
