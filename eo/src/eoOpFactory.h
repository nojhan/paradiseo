// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMonOpFactory.h
// (c) GeNeura Team, 1998
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

	/** Another factory method: creates an object from an istream, reading from
	it whatever is needed to create the object. Usually, the format for the istream will be\\
	objectType parameter1 parameter2 ... parametern\\
	If there are problems, an exception is raised; it should be caught at the
	upper level, because it might be something for that level
	@param _is an stream from where a single line will be read
	@throw runtime_exception if the object type is not known
	*/
	virtual eoOp<EOT>* make(istream& _is) {
		eoOp<EOT> * opPtr = NULL;
		string objectTypeStr;
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
	}


};


#endif _EOOPFACTORY_H
