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
#include <utility>

#include "edoRepairer.h"

/** Repair a candidate solution by sequentially applying several repairers on 
 * subparts of the solution (subparts being defined by the corresponding set 
 * of indexes).
 *
 * Only work on EOT that implements the "push_back( EOT::AtomType )" and 
 * "operator[](uint)" and "at(uint)" methods (i.e. random access containers).
 *
 * Expects _addresses_ of the repairer operators.
 *
 * Use the second template type if you want a different container to store
 * indexes. You can use any iterable. For example, you may want to use a set if
 * you need to be sure that indexes are use only once:
 *     edoRepairerDispatcher<EOT, std::set<unsigned int> > rpd; 
 *     std::set<unsigned int> idx(1,1);
 *     idx.insert(2);
 *     rpd.add( idx, &repairer );
 *
 * A diagram trying to visually explain how it works:
 \ditaa

       |
 /-\   |   /------------\
 | +---|---+ Dispatcher |
 | |   v   |            |
 | |+-----+| --------------------------------+
 | || x_0 ||  +-+-+-+   |   +------------\   |   /-\
 | |+-----+|  |2|3|5+*----*-* Repairer A +---|---+ |
 | || x_1 ||  +-+-+-+   | | |            |   v   | |
 | |+-----+|            | | |            |+-----+| |
 | || x_2 ||            | | |            || x_2 || |
 | |+-----+|            | | |            |+-----+| |
 | || x_3 ||            | | |            || x_3 || |
 | |+-----+|            | | |            |+-----+| |
 | || x_4 ||            | | |            || x_5 || |           
 | |+-----+|            | | |            |+-----+| |
 | || x_5 ||            | | |            |   |   | |
 | |+-----+|            | | |            +---|---+ |
 | || x_6 ||            | | \------------/   |   \-/
 | |+-----+| <-------------------------------+
 | || x_7 ||            | |
 | |+-----+|  +-+-+     | |
 | || x_8 ||  |2|3+*------+
 | |+-----+|  +-+-+     |
 | || x_9 ||            |
 | |+-----+|  +-+-+     |   +------------\       /-\
 | |   |   |  |1|5+*--------* Repairer B +-------+ |
 | |   |   |  +-+-+     |   |            |       | |
 | |   |   |            |   |            |       | |
 | |   |   |            |   |            +-------+ |
 | +---|---+            |   \------------/       \-/
 \-/   |   \------------/
       v

 \endditaa

 * @example t-dispatcher-round.cpp
 *
 * @ingroup Repairers
 */
template < typename EOT, typename ICT = std::vector<unsigned int> >
class edoRepairerDispatcher 
    : public edoRepairer<EOT>, 
             std::vector< 
                  std::pair< ICT, edoRepairer< EOT >* > 
             >
{
public:

    //! Empty constructor
    edoRepairerDispatcher() : 
        std::vector< 
            std::pair< std::vector< unsigned int >, edoRepairer< EOT >* > 
        >()
    {}

    //! Constructor with a single index set and repairer operator
    edoRepairerDispatcher( ICT idx, edoRepairer<EOT>* op ) :
        std::vector< 
            std::pair< std::vector< unsigned int >, edoRepairer< EOT >* > 
        >() 
    {
        this->add( idx, op );
    }

    //! Add more indexes set and their corresponding repairer operator address to the list
    void add( ICT idx, edoRepairer<EOT>* op )
    {
#ifndef NDEBUG
        if( idx.size() == 0 ) {
            eo::log << eo::warnings << "A repairer is added to the dispatcher while having an empty index list, nothing will be repaired" << std::endl;
        }
#endif
        assert( op != NULL );

        this->push_back( std::make_pair(idx, op) );
    }

    //! Repair a solution by calling several repair operator on subset of indexes
    virtual void operator()( EOT& sol )
    {
//        std::cout << "in dispatcher, sol = " << sol << std::endl;

        // iterate over { indexe, repairer }
        // ipair is an iterator that points on a pair of <indexes,repairer>
        for( typename edoRepairerDispatcher<EOT>::iterator ipair = this->begin(); ipair != this->end(); ++ipair ) {

            assert( ipair->first.size() <= sol.size() ); // assert there is less indexes than items in the whole solution

            // a partial copy of the sol
            EOT partsol;

//            std::cout << "\tusing indexes = ";
//
            // iterate over indexes
            // j is an iterator that points on an uint
            for( std::vector< unsigned int >::iterator j = ipair->first.begin(); j != ipair->first.end(); ++j ) {

//                std::cout << *j << " ";
//                std::cout.flush();

                partsol.push_back( sol.at(*j) );
            } // for j
//            std::cout << std::endl;
//            std::cout << "\tpartial sol = " << partsol << std::endl;

            if( partsol.size() == 0 ) {
                continue;
            }
            assert( partsol.size() > 0 );

            // apply the repairer on the partial copy
            // the repairer is a functor, thus second is callable
            (*(ipair->second))( partsol );

            { // copy back the repaired partial solution to sol
                // browse partsol with uint k, and the idx set with an iterator (std::vector is an associative tab)
                unsigned int k=0;
                for( std::vector< unsigned int >::iterator j = ipair->first.begin(); j != ipair->first.end(); ++j ) {
                    sol[ *j ] = partsol[ k ];
                    k++;
                } // for j
            } // context for k
        } // for ipair

        sol.invalidate();
    }
};

#endif // !_edoRepairerDispatcher_h
