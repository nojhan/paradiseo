// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoString.h
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

  	/** Ctor from a stream
	@param _s input stream
	*/
	eoString( istream & _s )
	  : eo1d<char, fitnessT>(){ 
	  _s >> *this;
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
	virtual char getGene( unsigned _i ) const {
		if ( _i >= length() ) 
			throw out_of_range( "out_of_range when reading gene");
		return (*this)[_i];
	};
	
	/** methods that implement the eo1d <em>protocol</em>
	    @exception out_of_range if _i is larger than EO큦 size
	*/
	virtual void setGene( unsigned _i, const char& _value )  {
	  if ( _i >= size() )
	    throw out_of_range( "out_of_range when writing a gene");
	  (*this)[_i] = _value;
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
	virtual string className() const {return "eoString";};
    //@}
	

};

#endif

