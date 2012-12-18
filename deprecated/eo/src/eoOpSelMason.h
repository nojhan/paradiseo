// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoOpSelMason.h
// (c) GeNeura Team, 1999
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

#ifndef _EOOPSELMASON_H
#define _EOOPSELMASON_H

#include <eoOpFactory.h>	// for eoFactory and eoOpFactory

#include <map>


/** EO Mason, or builder, for operator selectors. A builder must allocate memory
to the objects it builds, and then deallocate it when it gets out of scope

@ingroup Utilities
*/
template<class eoClass>
class eoOpSelMason: public eoFactory<eoOpSelector<eoClass> > {

public:
        typedef std::vector<eoOp<eoClass>* > vOpP;
        typedef map<eoOpSelector<eoClass>*, vOpP > MEV;

        /// @name ctors and dtors
        //{@
        /// constructor
        eoOpSelMason( eoOpFactory<eoClass>& _opFact): operatorFactory( _opFact ) {};

        /// destructor
        virtual ~eoOpSelMason() {};
        //@}

        /** Factory methods: creates an object from an std::istream, reading from
        it whatever is needed to create the object. The format is
        opSelClassName\\
        rate 1 operator1\\
        rate 2 operator2\\
        ...\\
        Stores all operators built in a database (#allocMap#), so that somebody
        can destroy them later. The Mason is in charge or destroying the operators,
        since the built object can´t do it itself. The objects built must be destroyed
        from outside, using the "destroy" method
        */
        virtual eoOpSelector<eoClass>* make(std::istream& _is) {

                std::string opSelName;
                _is >> opSelName;
                eoOpSelector<eoClass>* opSelectorP;
                // Build the operator selector
                if ( opSelName == "eoProportionalOpSel" ) {
                        opSelectorP = new eoProportionalOpSel<eoClass>();
                }

                // Temp std::vector for storing pointers
                vOpP tmpPVec;
                // read operator rate and name
                while ( _is ) {
                        float rate;
                        _is >> rate;
                        if ( _is ) {
                                eoOp<eoClass>* op = operatorFactory.make( _is );	// This reads the rest of the line
                                // Add the operators to the selector, don´t pay attention to the IDs
                                opSelectorP->addOp( *op, rate );
                                // Keep it in the store, to destroy later
                                tmpPVec.push_back( op );
                        } // if
                } // while

                // Put it in the map
                allocMap.insert( MEV::value_type( opSelectorP, tmpPVec ) );

                return opSelectorP;
        };

        ///@name eoObject methods
        //@{
        /** Return the class id */
        virtual std::string className() const { return "eoOpSelMason"; }

        //@}

private:
        map<eoOpSelector<eoClass>*,std::vector<eoOp<eoClass>* > > allocMap;
        eoOpFactory<eoClass>& operatorFactory;
};


#endif
