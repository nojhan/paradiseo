// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/*
-----------------------------------------------------------------------------
File............: eo2d.h
Author..........: Geneura Team (this file: Victor Rivas, vrivas@ujaen.es)
Date............: 21-Sep-1999, at Fac. of Sciences, Univ. of Granada (Spain)
Description.....: Implementation of a 2-dimensional chromosome.

  ================  Modif. 1  ================
  Author........:
  Date..........:
  Description...:

-----------------------------------------------------------------------------
*/
//-----------------------------------------------------------------------------
// eo2d.h 
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

#ifndef _EO2D_H
#define _EO2D_H

#include <iostream>				// for ostream
#include <vector>

// EO Includes
#include <EO.h>
#include <eoRnd.h>

using namespace std;

/** @name eo2d class 
* Randomly accesible  evolvable object with two dimension, with
variable length each of them.  
* Use this if you want to evolve "two-dimensional" things, like bit arrays, or
floating-point arrays. If you don't, subclass directly from EO
* @see EO
* @author GeNeura
* @version 0.2
*/

//@{

/** eo2d: Base class for "chromosomes" with a double dimension
#T# is the type it will be instantiated with; this type must have, at
least, a copy ctor, assignment operators, 
*/
template<class T, class fitnessT = float>
class eo2d: public EO< fitnessT > {
public:

  /// Declaration to make it accessible from subclasses
  typedef T Type;

  /** Can be used as default ctor; should be called from derived
      classes. Fitness should be  at birth
  */
  eo2d()
    :EO<fitnessT> ( ) {};

  /** Ctor using a random number generator and with an specified size
      @param _rows Initial number of rows
      @param _columns Initial number of columns
      @param _rndGen Random "T-type" generator
      @param _ID An ID string, preferably unique
  */
  eo2d( const unsigned _rows, 
	const unsigned _columns,
	eoRnd<T>& _rndGen, 
	const string& _ID = "");

 /** Ctor from an istream. It just passes the stream to EO, subclasses should
     have to implement this.
     @param _is the input stream
 */ 
  eo2d( istream& _is): EO<fitnessT>( _is ){};

  /// Copy ctor
  eo2d( const eo2d& _eo )
    :EO<fitnessT> ( _eo ) {};

  /// Assignment operator
  const eo2d& operator= ( const eo2d& _eo ) {
	  EO<fitnessT>::operator = ( _eo );
	  return *this;
  }

  /// Needed virtual dtor
  virtual ~eo2d(){};

  /** Reads and returns a copy of the gene in position _r,_c.\
      This implies that T must have a copy ctor .
      @param _r Index for rows. Must be an unsigned less than #numOfRows()#  
      @param _c Index for columns. Must be an unsigned less than #numOfCols()#
      @return what's inside the gene, with the correct type
      @exception out_of_range if _r >=numOfRows()
      @exception out_of_range if _c >=numOfCols()
    */
  virtual T getGene( const unsigned _r,
		     const unsigned _j ) const = 0;

  /** Overwrites the gene placed in position _r,_c with a
   * new value. This means that the assignment operator
   * for T must be defined .
   @param _r Index for rows. Must be an unsigned less than #numOfRows()#  
   @param _c Index for columns. Must be an unsigned less than #numOfCols()#
   @return what's inside the gene, with the correct type
   @exception out_of_range if _r >=numOfRows()
   @exception out_of_range if _c >=numOfCols()
  */
  virtual void setGene( const unsigned _r, 
			const unsigned _c, 
			const T& _value ) = 0;

  /** Inserts a row, moving the rest to the bottom. 
   * If _r = numOfRows(), it insert it at the end.
   * Obviously, changes number of rows. 
   @param _r Position where the new row will be inserted.
   @param _val Vector containing the new values to be inserted.
   @exception invalid_argument If _val has not numOfCols() components.
   @exception out_of_range If _r is greater than numOfRows()
  */
  virtual void insertRow( const unsigned _r, 
			  const vector<T>& _val ) = 0;

  /** Eliminates the row at position _r; all the other genes will
      be shifted up.
      @param _r Number of he row to be deleted.
      @exception out_of_range if _r >=numOfRows()
  */
  virtual void deleteRow( const unsigned _r ) = 0;

  /** Inserts a column, moving the rest to the right. 
   * If _c = numOfCols(), it insert it at the end.
   * Obviously, changes number of cols. 
   @param _r Position where the new column will be inserted.
   @param _val Vector containing the new values to be inserted.
   @exception invalid_argument if _val has not numOfRows() components.
  */
  virtual void insertCol( const unsigned _c, 
			  const vector<T>& _val ) = 0;

  /** Eliminates the column at position _c; all the other columns will
      be shifted left.
      @param _c Number of he column to be deleted.
      @exception out_of_range if _c >=numOfCols()
  */
  virtual void deleteCol( const unsigned _c ) = 0;
  
  /// Returns the number of rows in the eo2d
  virtual unsigned numOfRows() const = 0;

  /// Returns the number of columns in the eo2d
  virtual unsigned numOfCols() const = 0;

  /// @name Methods from eoObject
  //@{
  /**
   * Read object. Theoretically, the length is known in advance. All objects
   * Should call base class
   * @param _s A istream.
   * @throw runtime_exception If a valid object can't be read.
   */
  virtual void readFrom(istream& _s) {

    for ( unsigned r = 0; r < numOfRows(); ++r ) {
      for ( unsigned c = 0; c < numOfCols(); ++c ) {
	T tmp;
	_s >> tmp;
	setGene( r, c, tmp );
      }
    }
    // there is no way of distinguishing fitness from the object, so
    // it's skipped
  }

  /** Print itself: inherited from eoObject implementation. 
      Instance from base classes are processed in
      base classes, so you don´t have to worry about, for instance, fitness.
  @param _s the ostream in which things are written*/
  virtual void printOn( ostream& _s ) const{
    for ( unsigned r = 0; r < numOfRows(); ++r ) {
      for ( unsigned c = 0; c < numOfCols(); ++c ) {
	_s << getGene( r,c ) << " ";
      }
    }
  }

  /** Inherited from eoObject 
      @see eoObject
  */
  string className() const {return "eo2d";};

  //@}

};

//@}


// --------------- Implementations --------------------------

/** Ctor using a random number generator and with an specified size
    @param _rows Initial number of rows
    @param _columns Initial number of columns
    @param _rndGen Random "T-type" generator
    @param _ID An ID string, preferably unique
*/
template< class T, class fitnessT>
eo2d<T,fitnessT>::eo2d<T,fitnessT>( const unsigned _rows, 
				    const unsigned _columns,
				    eoRnd<T>& _rndGen, 
				    const string& _ID = "")
  :EO<fitnessT> ( _ID ) {
  for ( unsigned r = 0; r < _rows; ++r ) {
    for ( unsigned c = 0; c < _cols; ++c ) {
      insertGene( r, c, _rndGen() );
    }
  }
};


#endif
