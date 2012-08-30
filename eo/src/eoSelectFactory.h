// eoSelectFactory.h
// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// EOFactory.h
// (c) GeNeura Team, 1998
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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */
//-----------------------------------------------------------------------------

#ifndef _EOSELECTFACTORY_H
#define _EOSELECTFACTORY_H

#include <eoFactory.h>
#include <eoRandomSelect.h>
#include <eoTournament.h>

//-----------------------------------------------------------------------------

/** EO Factory.An instance of the factory class to create selectors, that is,
eoSelect objects

@see eoSelect
@ingroup Selectors
@ingroup Utilities
*/
template< class EOT>
class eoSelectFactory: public eoFactory<eoSelect< EOT> > {

public:

        /// @name ctors and dtors
        //{@
        /// constructor
        eoSelectFactory( ) {}

        /// destructor
        virtual ~eoSelectFactory() {}
        //@}

        /** Another factory methods: creates an object from an std::istream, reading from
        it whatever is needed to create the object. Usually, the format for the std::istream will be\\
        objectType parameter1 parameter2 ... parametern\\
        */
        virtual eoSelect<EOT>* make(std::istream& _is) {
                eoSelect<EOT> * selectPtr;
                std::string objectTypeStr;
                _is >> objectTypeStr;
                // All selectors have a rate, the proportion of the original population
                float rate;
                _is >> rate;
                if  ( objectTypeStr == "eoTournament") {
                        // another parameter is necessary
                        unsigned tSize;
                        _is >> tSize;
                        selectPtr = new eoTournament<EOT>( rate, tSize );
                } else  {
                        if ( objectTypeStr == "eoRandomSelect" ) {
                                selectPtr = new eoRandomSelect<EOT>( rate );
                        } else {
                                                throw std::runtime_error( "Incorrect selector type" );
                        }
                }
                return selectPtr;
        }

        ///@name eoObject methods
        //@{
        void printOn( std::ostream& _os ) const {};
        void readFrom( std::istream& _is ){};

        /** className is inherited */
        //@}

};


#endif
