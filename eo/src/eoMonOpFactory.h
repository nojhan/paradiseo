// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMonOpFactory.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOMONOPFACTORY_H
#define _EOMONOPFACTORY_H

#include <eoFactory.h>
#include <eoDup.h>
#include <eoKill.h>
#include <eoTranspose.h>

//-----------------------------------------------------------------------------

/** EO Factory. An instance of the factory class to create monary operators.
@see eoSelect*/
template< class EOT>
class eoMonOpFactory: public eoFactory< eoMonOp<EOT> >  {
	
public:
	
	/// @name ctors and dtors
	//{@
	/// constructor
	eoMonOpFactory( ) {}
	
	/// destructor
	virtual ~eoMonOpFactory() {}
	//@}

	/** Another factory method: creates an object from an istream, reading from
	it whatever is needed to create the object. Usually, the format for the istream will be\\
	objectType parameter1 parameter2 ... parametern\\
	*/
	virtual eoMonOp<EOT>* make(istream& _is) {
		eoMonOp<EOT> * opPtr;
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
		if ( !opPtr ) {
		  throw runtime_error( "Incorrect selector type" );
		}
		return opPtr;
	}

	///@name eoObject methods
	//@{
	void printOn( ostream& _os ) const {};
	void readFrom( istream& _is ){};

	/** className is inherited */
	//@}
	
};


#endif _EOFACTORY_H
