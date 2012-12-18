// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFactory.h
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

#ifndef _EOFACTORY_H
#define _EOFACTORY_H

//-----------------------------------------------------------------------------
#include <eoObject.h>

//-----------------------------------------------------------------------------

/** EO Factory. A factory is used to create other objects. In particular,
it can be used so that objects of that kind can´t be created in any other
way. It should be instantiated with anything that needs a factory, like selectors
or whatever; but the instance class should be the parent class from which all the
object that are going to be created descend. This class basically defines an interface,
as usual. The base factory class for each hierarchy should be redefined every time a new
object is added to the hierarchy, which is not too good, but in any case, some code would
have to be modified

@ingroup Utilities
*/
template<class EOClass>
class eoFactory: public eoObject {

public:

        /// @name ctors and dtors
        //{@
        /// constructor
        eoFactory( ) {}

        /// destructor
        virtual ~eoFactory() {}
        //@}

        /** Another factory methods: creates an object from an std::istream, reading from
        it whatever is needed to create the object. Usually, the format for the std::istream will be\\
        objectType parameter1 parameter2 ... parametern\\
        */
        virtual EOClass* make(std::istream& _is) = 0;

        ///@name eoObject methods
        //@{
        /** Return the class id */
        virtual std::string className() const { return "eoFactory"; }

        /** Read and print are left without implementation */
        //@}

};


#endif
