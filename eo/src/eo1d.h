// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eo1d.h 
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

#ifndef _EO1D_H
#define _EO1D_H

#include <iostream>				// for ostream

// EO Includes
#include <EO.h>
#include <eoRnd.h>

using namespace std;

/** @name eo1d class 
* Randomly accesible  evolvable object with one dimension, with
variable length.  
* Use this if you want to evolve "linear" things, like bitstrings, or
floating-point arrays. If you don't, subclass directly from EO
* @see EO
* @author GeNeura
* @version 0.2
*/

//@{

/** eo1d: Base class for "chromosomes" with a single dimension
#T# is the type it will be instantiated with; this type must have, at
least, a copy ctor, assignment operators, 
*/
template<class T, class fitnessT = float>
class eo1d: public EO< fitnessT > {
public:

  /// Declaration to make it accessible from subclasses
  typedef T Type;

  /** Can be used as default ctor; should be called from derived
      classes. Fitness should be  at birth
  */
  eo1d()
    :EO<fitnessT> ( ) {};

  /** Ctor using a random number generator and with an specified size
      @param _rndGen Random number generator
      @param _size lineal dimension of the eo1d
      @param _ID An ID string, preferably unique
  */
  eo1d( unsigned _size, eoRnd<T>& _rndGen, const string& _ID = "");

 /** Ctor from a istream. It just passes the stream to EO, subclasses should
     have to implement this.
     @param _is the input stream
 */ 
  eo1d( istream& _is): EO<fitnessT>( _is ){};

  /// Copy ctor
  eo1d( const eo1d& _eo )
    :EO<fitnessT> ( _eo ) {};

  /// Assignment operator
  const eo1d& operator= ( const eo1d& _eo ) {
	  EO<fitnessT>::operator = ( _eo );
	  return *this;
  }

  /// Needed virtual dtor
  virtual ~eo1d(){};

  /** Reads and returns a copy of the gene in position _i
      This implies that T must have a copy ctor .
      @param _i index of the gene, which is the minimal unit. Must be
      an unsigned less than #length()#  
      @return what's inside the gene, with the correct type
	  @exception out_of_range if _i > size()
    */
  virtual T getGene( unsigned _i ) const = 0;

  /** Overwrites the gene placed in position _i with a
   * new value. This means that the assignment operator
   * for T must be defined .
   @param _i index
   @return what's inside the gene, with the correct type
   @exception out_of_range if _i > size()
  */
  virtual void setGene( unsigned _i, const T& _value ) = 0;

  /** Inserts a gene, moving the rest to the right. If
   * _i = length(), it should insert it at the end.
   * Obviously, changes length 
   @param _i index
   @param _val new value
  */
  virtual void insertGene( unsigned _i, T _val ) = 0;

  /** Eliminates the gene at position _i; all the other genes will
      be shifted left
      @param _i index of the gene that is going to be modified.
  */
  virtual void deleteGene( unsigned _i ) = 0;

  /// Returns the number of genes in the eo1d
  virtual unsigned length() const = 0;

  /// @name Methods from eoObject
  //@{
  /**
   * Read object. Theoretically, the length is known in advance. All objects
   * Should call base class
   * @param _s A istream.
   * @throw runtime_exception If a valid object can't be read.
   */
  virtual void readFrom(istream& _s) {

    for ( unsigned i = 0; i < length(); i ++ ) {
      T tmp;
      _s >> tmp;
      setGene( i, tmp );
    }
    // there is no way of distinguishing fitness from the object, so
    // it's skipped
  }

  /** Print itself: inherited from eoObject implementation. 
      Instance from base classes are processed in
      base classes, so you don´t have to worry about, for instance, fitness.
  @param _s the ostream in which things are written*/
  virtual void printOn( ostream& _s ) const{
    for ( unsigned i = 0; i < length(); i ++ ) {
      _s << getGene( i ) << " ";
    }
  }

  /** Inherited from eoObject 
      @see eoObject
  */
  string className() const {return "eo1d";};

  //@}

};

//@}


// --------------- Implementations --------------------------

/* Ctor using a random number generator and with an specified size
   @param _rndGen Random number generator
   @param _size lineal dimension of the eo1d
   @param _ID An ID string, preferably unique
*/
template< class T, class fitnessT>
eo1d<T,fitnessT>::eo1d<T,fitnessT>( unsigned _size, eoRnd<T>& _rndGen,
				    const string& _ID )
	:EO<fitnessT> ( _ID ) {
  for ( unsigned i = 0; i < _size; i ++ ) {
    insertGene( i, _rndGen() );
  }
};


#endif
