/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoVector.h
       Turns an STL std::vector into an EO
   (c) GeNeura Team, 1998
 
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

#ifndef _eoVector_H
#define _eoVector_H

// STL libraries
#include <vector>		// For std::vector<int>
#include <stdexcept>
#include <strstream>

#include <eo1d.h>
#include <eoRnd.h>

/** Adaptor that turns an STL std::vector into an EO
 with the same gene type as the type with which
 the std::vector has been instantiated
*/
template <class T, class fitnessT=float>
class eoVector: public eo1d<T, fitnessT>, public std::vector<T> {
public:
  typedef T Type ;
  
  /// Canonical part of the objects: several ctors, copy ctor, dtor and assignment operator
  //@{
  
  /** Ctor. 
      @param _size Lineal length of the object
      @param _val Common initial value
  */
  eoVector( unsigned _size = 0, T _val = 0)
    : eo1d<T, fitnessT>(), std::vector<T>( _size, _val ){ };
  
  /** Ctor using a random number generator
      @param _size Lineal length of the object
      @param _rnd a random number generator, which returns a random value each time it큦 called
  */
  eoVector( unsigned _size, eoRnd<T>& _rnd );

  /** Ctor from a std::istream. The T class should accept reading from a std::istream. It doesn't read fitness,
which is supposed to be dynamic and dependent on environment. 
      @param _is the input stream; should have all values in a single line, separated by whitespace
  */
  eoVector( std::istream& _is);
  

  /// copy ctor
  eoVector( const eoVector & _eo )
    : eo1d<T, fitnessT>( _eo ), std::vector<T>( _eo ){ };
  
  /// Assignment operator
  const eoVector& operator =( const eoVector & _eo ) {
    if ( this != &_eo ){
      eo1d<T, fitnessT>::operator=( _eo );
      std::vector<T>::operator=( _eo );
    }
    return *this;
  }
  
  /// dtor
  virtual ~eoVector() {};
  
  //@}
  
  /** methods that implement the eo1d <em>protocol</em>
      @std::exception out_of_range if _i is larger than EO큦 size
  */
  virtual T getGene( unsigned _i ) const {
    if ( _i >= length() ) 
      throw out_of_range( "out_of_range when reading gene");
    return (*this)[_i];
  };
  
  /** methods that implement the eo1d <em>protocol</em>
      @std::exception out_of_range if _i is larger than EO큦 size
  */
  virtual void setGene( unsigned _i, const T& _value ) {
    if ( _i >= size() )
      throw out_of_range( "out_of_range when writing a gene");
    (*this)[_i] = _value;
  };
  
  /** methods that implement the eo1d <em>protocol</em>
      @std::exception out_of_range if _i is larger than EO큦 size
  */
  virtual void insertGene( unsigned _i, T _val ) {
    if (_i <= size() ) {
      std::vector<T>::iterator i = begin()+_i;
      insert( i, _val );
    } else {
      throw out_of_range( "out_of_range when inserting a gene");
    }
  };

  /** Eliminates the gene at position _i
      @std::exception out_of_range if _i is larger than EO큦 size
  */
  virtual void deleteGene( unsigned _i ) { 
    if (_i < this->size() ) {
      std::vector<T>::iterator i = this->begin()+_i;
      this->erase( i );
    } else {
      throw out_of_range( "out_of_range when deleting a gene");
    };
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
  std::string className() const {return "eoVector";};
  //@}
	
};


//____________________________ Some method implementation _______________________

// Ctors______________________________________________________________________________
//____________________________________________________________________________________
template <class T, class fitnessT>
eoVector<T,fitnessT>::eoVector( unsigned _size, eoRnd<T>& _rnd )
  : eo1d<T, fitnessT>(), std::vector<T>( _size ){ 
  for ( iterator i = begin(); i != end(); i ++ ) {
    *i = _rnd();
  }
};

//____________________________________________________________________________________
template <class T, class fitnessT>
eoVector<T,fitnessT>::eoVector( std::istream& _is)
  : eo1d<T, fitnessT>(), std::vector<T>( ){ 
  while (_is ) {
    T tmp;
    _is >> tmp;
    push_back( tmp );
  }

};

#endif
