// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoString.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _eoString_H
#define _eoString_H

// STL libraries
#include <string>		
#include <stdexcept>

using namespace std;

// EO headers
#include <eo1d.h>

//-----------------------------------------------------------------------------
// eoString
//-----------------------------------------------------------------------------

/** Adaptor that turns an STL string into an EO */
template <class fitnessT >
class eoString: public eo1d<char, fitnessT>, public string {
public:

	/// Canonical part of the objects: several ctors, copy ctor, dtor and assignment operator
	//@{
	/// ctor
	eoString( const string& _str ="" )
		: eo1d<char, fitnessT>(), string( _str ) {};

	
	/** Ctor using a random number generator
	@param _size Lineal length of the object
	@param _rnd a random number generator, which returns a random value each time it큦 called
	*/
	eoString( unsigned _size, eoRnd<char>& _rnd )
		: eo1d<char, fitnessT>(), string(){ 
		for ( unsigned i = 0; i < _size; i ++ ) {
			*this += _rnd();
		}
	};

	/// copy ctor
	eoString( const eoString<fitnessT>& _eoStr )
		:eo1d<char, fitnessT>( static_cast<const eo1d<char, fitnessT> & > ( _eoStr ) ), 
	  string( _eoStr ){};

	/// Assignment operator
	const eoString& operator =( const eoString& _eoStr ) {
		if ( this != & _eoStr ) {
			eo1d<char, fitnessT>::operator = ( _eoStr );
			string::operator = ( _eoStr );
		}
		return *this;
	}

	/// dtor
	virtual ~eoString() {};
//@}


	/** methods that implement the eo1d <em>protocol</em>
	    @exception out_of_range if _i is larger than EO큦 size
	*/
	virtual char gene( unsigned _i ) const {
		if ( _i >= length() ) 
			throw out_of_range( "out_of_range when reading gene");
		return (*this)[_i];
	};
	
	/** methods that implement the eo1d <em>protocol</em>
	    @exception out_of_range if _i is larger than EO큦 size
	*/
	virtual char& gene( unsigned _i )  {
	  if ( _i >= size() )
	    throw out_of_range( "out_of_range when writing a gene");
	  return (*this)[_i];
	};

	/** Inserts a value after _i, displacing anything to the right
	 @exception out_of_range if _i is larger than EO큦 size
	 */
	virtual void insertGene( unsigned _i, char _val ) {
		if (_i <= this->size() ) {
			string::iterator i = this->begin()+_i;
			this->insert( i, _val );
		} else 
			throw out_of_range( "out_of_range when inserting gene");
	};

	/** Eliminates the gene at position _i
	 @exception out_of_range if _i is larger than EO큦 size
	 */
	virtual void deleteGene( unsigned _i ) { 
	  if (_i < this->size() ) {
	    string::iterator i = this->begin()+_i;
	    this->erase( i );
	  } else 
	    throw out_of_range( "out_of_range when deleting gene");
	};
	
	/// methods that implement the EO <em>protocol</em>
	virtual unsigned length() const { return this->size(); };

	/** @name Methods from eoObject
	readFrom and printOn are directly inherited from eo1d
	*/
	//@{
	/** Inherited from eoObject 
		  @see eoObject
	*/
	string className() const {return "eoString";};
    //@}
	

};

#endif
