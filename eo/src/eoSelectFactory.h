// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// EOFactory.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOSELECTFACTORY_H
#define _EOSELECTFACTORY_H

#include <eoFactory.h>
#include <eoRandomSelect.h>
#include <eoTournament.h>

//-----------------------------------------------------------------------------

/** EO Factory.An instance of the factory class to create selectors, that is,
eoSelect objects
@see eoSelect*/
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

	/** Another factory methods: creates an object from an istream, reading from
	it whatever is needed to create the object. Usually, the format for the istream will be\\
	objectType parameter1 parameter2 ... parametern\\
	*/
	virtual eoSelect<EOT>* make(istream& _is) {
		eoSelect<EOT> * selectPtr;
		string objectTypeStr;
		_is >> objectTypeStr;
		// All selectors have a rate, the proportion of the original population
		float rate;
		_is >> rate;
		if  ( objectTypeStr == "eoTournament") {
			// another parameter is necessary
			unsigned tSize;
			_is >> tSize;
			selectPtr = new eoTournament<EOT>( rate, tSize );
		} else 	{
			if ( objectTypeStr == "eoRandomSelect" ) {
				selectPtr = new eoRandomSelect<EOT>( rate );
			} else {
						throw runtime_error( "Incorrect selector type" );
			}
		}
		return selectPtr;
	}

	///@name eoObject methods
	//@{
	void printOn( ostream& _os ) const {};
	void readFrom( istream& _is ){};

	/** className is inherited */
	//@}
	
};


#endif _EOFACTORY_H
