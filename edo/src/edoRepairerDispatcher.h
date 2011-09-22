/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2011 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Pierre Savéant <pierre.saveant@thalesgroup.com>
*/

#ifndef _edoRepairerDispatcher_h
#define _edoRepairerDispatcher_h

#include <vector>
#include <set>
#include <utility>

#include "edoRepairer.h"

/** Repair a candidate solution by sequentially applying several repairers on 
 * subparts of the solution (subparts being defined by the corresponding set 
 * of indexes).
 *
 * Only work on EOT that implements the "push_back( EOT::AtomType )" and 
 * "operator[](uint)" and "at(uint)" methods.
 *
 * Expects _addresses_ of the repairer operators.
 *
 * @example t-dispatcher-round.cpp
 *
 * @ingroup Repairers
 */

template < typename EOT >
class edoRepairerDispatcher 
    : public edoRepairer<EOT>, 
             std::vector< 
                  std::pair< std::set< unsigned int >, edoRepairer< EOT >* > 
             >
{
public:
    //! Empty constructor
    edoRepairerDispatcher() : 
        std::vector< 
            std::pair< std::set< unsigned int >, edoRepairer< EOT >* > 
        >()
    {}

    //! Constructor with a single index set and repairer operator
    edoRepairerDispatcher( std::set<unsigned int> idx, edoRepairer<EOT>* op ) :
        std::vector< 
            std::pair< std::set< unsigned int >, edoRepairer< EOT >* > 
        >() 
    {
        this->add( idx, op );
    }

    //! Add more indexes set and their corresponding repairer operator address to the list
    void add( std::set<unsigned int> idx, edoRepairer<EOT>* op )
    {
        assert( idx.size() > 0 );
        assert( op != NULL );

        this->push_back( std::make_pair(idx, op) );
    }

    //! Repair a solution by calling several repair operator on subset of indexes
    virtual void operator()( EOT& sol )
    {
        // ipair is an iterator that points on a pair
        for( typename edoRepairerDispatcher<EOT>::iterator ipair = this->begin(); ipair != this->end(); ++ipair ) {
            // a partial copy of the sol
            EOT partsol;

            // j is an iterator that points on an uint
            for( std::set< unsigned int >::iterator j = ipair->first.begin(); j != ipair->first.end(); ++j ) {
                partsol.push_back( sol.at(*j) );
            } // for j

            assert( partsol.size() > 0 );

            // apply the repairer on the partial copy
            // the repairer is a functor, thus second is callable
            (*(ipair->second))( partsol );

            { // copy back the repaired partial solution to sol
                // browse partsol with uint k, and the idx set with an iterator (std::set is an associative tab)
                unsigned int k=0;
                for( std::set< unsigned int >::iterator j = ipair->first.begin(); j != ipair->first.end(); ++j ) {
                    sol[ *j ] = partsol[ k ];
                    k++;
                } // for j
            } // context for k
        } // for ipair
    }
};

#endif // !_edoRepairerDispatcher_h
