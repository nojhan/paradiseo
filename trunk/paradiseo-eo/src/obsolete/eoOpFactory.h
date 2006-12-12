// eoOpFactory.h
// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMonOpFactory.h
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

#ifndef _EOOPFACTORY_H
#define _EOOPFACTORY_H

#include <eoFactory.h>
#include <eoDup.h>
#include <eoKill.h>
#include <eoTranspose.h>
#include <eoXOver2.h>

//-----------------------------------------------------------------------------

/** EO Factory. An instance of the factory class to create monary operators.
@see eoSelect*/
template< class EOT>
class eoOpFactory: public eoFactory< eoOp<EOT> >  {
	
public:
	
	/// @name ctors and dtors
	//{@
	/// constructor
	eoOpFactory( ) {}
	
	/// destructor
	virtual ~eoOpFactory() {}
	//@}

	/** Another factory method: creates an object from an std::istream, reading from
	it whatever is needed to create the object. Usually, the format for the std::istream will be\\
	objectType parameter1 parameter2 ... parametern\\
	If there are problems, an std::exception is raised; it should be caught at the
	upper level, because it might be something for that level
	@param _is an stream from where a single line will be read
	@throw runtime_std::exception if the object type is not known
	*/
	virtual eoOp<EOT>* make(std::istream& _is) {
		eoOp<EOT> * opPtr = NULL;
		std::string objectTypeStr;
		_is >> objectTypeStr;
		if  ( objectTypeStr == "eoDup") {
		  opPtr = new eoDup<EOT>();
		} 
		if ( objectTypeStr == "eoKill" ) {
		  opPtr = new eoKill<EOT>( );
		} 
		if ( objectTypeStr == "eoTranspose" ) {
		  opPtr = new eoTranspose<EOT>( );
		} 
		if ( objectTypeStr == "eoXOver2" ) {
		  opPtr = new eoXOver2<EOT>( );
		} 
		if ( !opPtr ) {
		  throw objectTypeStr;
		}
		return opPtr;
	};


};


#endif _EOOPFACTORY_H
