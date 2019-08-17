/*
 <moNKlandscapesBitsIncrEval.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010
 
 Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau
 
 This software is governed by the CeCILL license under French law and
 abiding by the rules of distribution of free software.  You can  ue,
 modify and/ or redistribute the software under the terms of the CeCILL
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".
 
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

#ifndef _moNKlandscapesBitsIncrEval_H
#define _moNKlandscapesBitsIncrEval_H

#include "../../eval/moEval.h"
#include "../../../problems/eval/nkLandscapesEval.h"
#include <vector>

/**
 *
 * Incremental evaluation function (1 bit flip, Hamming distance 1)
 * for the NK-landscapes problem
 *
 *
 */
template< class Neighbor >
class moNKlandscapesBitsIncrEval : public moEval<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;
    
    /*
     * Constructor
     *
     * @param _nk fitness function of the NK landscapes
     */
    moNKlandscapesBitsIncrEval(nkLandscapesEval<EOT> & _nk) : nk(_nk) {
        inverseLinks = new std::vector<unsigned>[ nk.N ];
        
        // compute the contributions which are modified by flipping one bit
        for(unsigned int i = 0; i < nk.N; i++)
            for(unsigned int j = 0; j < nk.K + 1; j++) {
                inverseLinks[ nk.links[i][j] ].push_back(i);
            }
    }
    
    /*
     * Destructor
     *
     */
    ~moNKlandscapesBitsIncrEval() {
        delete [] inverseLinks;
    }
    
    /*
     * incremental evaluation of the neighbor for the oneMax problem
     * @param _solution the solution to move (bit string)
     * @param _neighbor the neighbor to consider (of type moBitNeigbor)
     */
    virtual void operator()(EOT & _solution, Neighbor & _neighbor) {
        unsigned int b;

        unsigned sig, nonSig;
        unsigned i;
        
        double delta = 0 ;
        
        for(unsigned k = 0; k < _neighbor.nBits; k++) {
            b = _neighbor.bits[k];

            for(unsigned int j = 0; j < inverseLinks[b].size(); j++) {
                i = inverseLinks[b][j];
                sigma(_solution, i, b, sig, nonSig);
                delta += nk.tables[i][nonSig] - nk.tables[i][sig];
            }
            
            // move the solution on this bit
            _solution[b] = !_solution[b];
        }

        // move back the solution
        for(unsigned k = 0; k < _neighbor.nBits; k++) {
            b = _neighbor.bits[k];
            _solution[b] = !_solution[b];
        }
        //std::cout << delta << std::endl;
        
        _neighbor.fitness(_solution.fitness() + delta / (double) nk.N);
    }
    
private:
    // Original nk fitness function
    nkLandscapesEval<EOT> & nk;
    
    // give the list of contributions which are modified when the corresponding bit is flipped
    std::vector<unsigned> * inverseLinks;
    
    /**
     * Compute the mask of the linked bits, and the mask when the bit is flipped
     *
     * @param _solution the solution to evaluate
     * @param i the bit of the contribution
     * @param _bit the bit to flip
     * @param sig value of the mask of contribution i
     * @param nonSig value of the mask of contribution i when the bit _bit is flipped
     */
    void sigma(EOT & _solution, int i, unsigned _bit, unsigned & sig, unsigned & nonSig) {
        sig    = 0;
        nonSig = 0;
        
        unsigned int n = 1;
        for(int j = 0; j < nk.K + 1; j++) {
            if (_solution[ nk.links[i][j] ] == 1)
                sig = sig | n;
            
            if (nk.links[i][j] == _bit)
                nonSig = n;
            
            n = n << 1;
        }
        
        nonSig = sig ^ nonSig;
    }
    
};

#endif

