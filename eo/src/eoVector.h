// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoVector.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------


#ifndef _eoVector_H
#define _eoVector_H

// STL libraries
#include <vector>		// For vector<int>
#include <stdexcept>
#include <strstream>

#include <eo1d.h>
#include <eoRnd.h>

/** Adaptor that turns an STL vector into an EO
 with the same gene type as the type with which
 the vector has been instantiated
*/
template <class T, class fitnessT>
class eoVector: public eo1d<T, fitnessT>, public vector<T> {
public:
  typedef Type T;
  
  /// Canonical part of the objects: several ctors, copy ctor, dtor and assignment operator
  //@{
  
  /** Ctor. 
      @param _size Lineal length of the object
      @param _val Common initial value
  */
  eoVector( unsigned _size = 0, T _val = 0)
    : eo1d<T, fitnessT>(), vector<T>( _size, _val ){ };
  
  /** Ctor using a random number generator
      @param _size Lineal length of the object
      @param _rnd a random number generator, which returns a random value each time it큦 called
  */
  eoVector( unsigned _size, eoRnd<T>& _rnd );

  /** Ctor from a istream. The T class should accept reading from a istream. It doesn't read fitness,
which is supposed to be dynamic and dependent on environment. 
      @param _is the input stream; should have all values in a single line, separated by whitespace
  */
  eoVector( istream& _is);
  

  /// copy ctor
  eoVector( const eoVector & _eo )
    : eo1d<T, fitnessT>( _eo ), vector<T>( _eo ){ };
  
  /// Assignment operator
  const eoVector& operator =( const eoVector & _eo ) {
    if ( this != &_eo ){
      eo1d<T, fitnessT>::operator=( _eo );
      vector<T>::operator=( _eo );
    }
    return *this;
  }
  
  /// dtor
  virtual ~eoVector() {};
  
  //@}
  
  /** methods that implement the eo1d <em>protocol</em>
      @exception out_of_range if _i is larger than EO큦 size
  */
  virtual T gene( unsigned _i ) const {
    if ( _i >= length() ) 
      throw out_of_range( "out_of_range when reading gene");
    return (*this)[_i];
  };
  
  /** methods that implement the eo1d <em>protocol</em>
      @exception out_of_range if _i is larger than EO큦 size
  */
  virtual T& gene( unsigned _i )  {
    if ( _i >= size() )
      throw out_of_range( "out_of_range when writing a gene");
    return operator[](_i);
  };
  
  /** methods that implement the eo1d <em>protocol</em>
      @exception out_of_range if _i is larger than EO큦 size
  */
  virtual void insertGene( unsigned _i, T _val ) {
    if (_i <= size() ) {
      vector<T>::iterator i = begin()+_i;
      insert( i, _val );
    } else {
      throw out_of_range( "out_of_range when inserting a gene");
    }
  };

  /** Eliminates the gene at position _i
      @exception out_of_range if _i is larger than EO큦 size
  */
  virtual void deleteGene( unsigned _i ) { 
    if (_i < this->size() ) {
      vector<T>::iterator i = this->begin()+_i;
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
  string className() const {return "eoVector";};
  //@}
	
};


//____________________________ Some method implementation _______________________

// Ctors______________________________________________________________________________
//____________________________________________________________________________________
template <class T, class fitnessT>
eoVector<T,fitnessT>::eoVector( unsigned _size, eoRnd<T>& _rnd )
  : eo1d<T, fitnessT>(), vector<T>( _size ){ 
  for ( iterator i = begin(); i != end(); i ++ ) {
    *i = _rnd();
  }
};

//____________________________________________________________________________________
template <class T, class fitnessT>
eoVector<T,fitnessT>::eoVector( istream& _is)
  : eo1d<T, fitnessT>(), vector<T>( ){ 
  while (_is ) {
    T tmp;
    _is >> tmp;
    push_back( tmp );
  }

};

#endif
