// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-
/*
-----------------------------------------------------------------------------
File............: eo2dVector.h
Author..........: Geneura Team (this file: Victor Rivas, vrivas@ujaen.es)
Date............: 29-Sep-1999, at Fac. of Sciences, Univ. of Granada (Spain)
Description.....: Implementation of a 2-dimensional chromosome usign STL 
                  std::vectors.

  ================  Modif. 1  ================
  Author........:
  Date..........:
  Description...:

QUEDA: Operador de asignación, lectura desde std::istream, escritura a std::ostream
-----------------------------------------------------------------------------
*/
//-----------------------------------------------------------------------------
// eo2dVector.h
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


#ifndef _eo2dVector_H
#define _eo2dVector_H

// STL libraries
#include <vector>		// For std::vector<int>
#include <stdexcept>
#include <strstream>
#include <ostream.h>

#include <eo2d.h>
#include <eoRnd.h>

/** Adaptor that turns an STL std::vector of vectror into an EO
    with the same gene type as the type with which
    the std::vector of std::vector has been instantiated.
*/
template <class T, class fitnessT>
class eo2dVector: public eo2d<T, fitnessT>, public std::vector< std::vector<T> > {
public:
  typedef T Type ;
  
  /** @name Canonical part of the objects: several ctors, copy ctor, \
   * dtor and assignment operator.
   */
  //@{
  
  /** Ctor. 
      @param _rows Number of rows.
      @param _cols Number of columns.
      @param _val Common initial value
  */
  eo2dVector( const unsigned _rows = 0, 
	      const unsigned _cols = 0, 
	      T _val = T() )
    : eo2d<T, fitnessT>(), std::vector< std::vector<T> >( _rows, std::vector<T>( _cols, _val ) ){};
  
  /** Ctor using a random number generator.
      @param _rows Number of rows.
      @param _cols Number of columns.
      @param _rnd A random "T-type" generator, which returns a random value each time it´s called.
  */
  eo2dVector( const unsigned _rows, 
	      const unsigned _cols, 
	      eoRnd<T>& _rnd );
  
  /** Ctor from a std::istream. The T class should accept reading from a std::istream. It doesn't read fitness,
      which is supposed to be dynamic and dependent on environment. 
      @param _is the input stream; should have all values in a single line, separated by whitespace
  */
  //eo2dVector( std::istream& _is);
  
  
  /// copy ctor
  eo2dVector( const eo2dVector & _eo )
    : eo2d<T, fitnessT>( _eo ), std::vector< std::vector<T> >( _eo ){ };
  
  /// Assignment operator
  /*
    const eo2dVector& operator =( const eo2dVector & _eo ) {
    if ( this != &_eo ){
    eo2d<T, fitnessT>::operator=( _eo );
    std::vector< <std::vector<T> >::operator=( _eo );
    }
    return *this;
    }
  */
  /// dtor
  virtual ~eo2dVector() {};
  
  //@}
  /** Reads and returns a copy of the gene in position _r,_c.\
      This implies that T must have a copy ctor .
      @param _r Index for rows. Must be an unsigned less than #numOfRows()#  
      @param _c Index for columns. Must be an unsigned less than #numOfCols()#
      @return what's inside the gene, with the correct type
      @std::exception out_of_range if _r >=numOfRows()
      @std::exception out_of_range if _c >=numOfCols()
  */
  virtual T getGene( const unsigned _r,
		     const unsigned _c ) const {
    if ( _r >= numOfRows() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::getGene: row out of range. " 
	  << "It should be <" << numOfRows() << '\0' << std::endl;
      throw out_of_range( msg.str() );
    }
    if ( _c >= numOfCols() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::getGene: column out of range. " 
	  << "It should be <" << numOfCols() << '\0' << std::endl;
      throw out_of_range( msg.str() );
    }
    return (*this)[_r][_c];
  };
  /** Overwrites the gene placed in position _r,_c with a
   * new value. This means that the assignment operator
   * for T must be defined .
   @param _r Index for rows. Must be an unsigned less than #numOfRows()#  
   @param _c Index for columns. Must be an unsigned less than #numOfCols()#
   @return what's inside the gene, with the correct type
   @std::exception out_of_range if _r >=numOfRows()
   @std::exception out_of_range if _c >=numOfCols()
  */
  virtual void setGene( const unsigned _r, 
			const unsigned _c, 
			const T& _value ) {
    if ( _r >= numOfRows() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::setGene: row out of range. " 
	  << "It should be <" << numOfRows() << '\0' << std::endl;
      throw out_of_range( msg.str() );
    }
    if ( _c >= numOfCols() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::setGene: column out of range. " 
	  << "It should be <" << numOfCols() << '\0' << std::endl;
      throw out_of_range( msg.str() );
    }
    (*this)[_r][_c]=_value;
  };
  
  
  
  /** Inserts a row, moving the rest to the bottom. 
   * If _r = numOfRows(), it insert it at the end.
   * Obviously, changes number of rows. 
   @param _r Position where the new row will be inserted.
   @param _val Vector containing the new values to be inserted.
   @std::exception invalid_argument If _val has not numOfCols() components.
   @std::exception out_of_range If _r is greater than numOfRows()
  */
  virtual void insertRow( const unsigned _r, 
			  const std::vector<T>& _val ) {
    // Test errors.
    if ( _r > numOfRows() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::insertRow: row out of range. " 
	  << "It should be <=" << numOfRows() << '\0' << std::endl;
      throw out_of_range( msg.str() );
    }    
    if ( _val.size() != numOfCols() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::insertRow: "
	  << "Incorrect number of values to be added. " 
	  << "It should be ==" << numOfCols() << '\0' << std::endl;
      throw invalid_argument( msg.str() );
    }
    
    // Insert the row.
    std::vector< std::vector<T> >::iterator ite = begin()+_r;
    insert( ite, _val );
  };
  
  /** Eliminates the row at position _r; all the other genes will
      be shifted up.
      @param _r Number of he row to be deleted.
      @std::exception out_of_range if _r >=numOfRows()
  */
  virtual void deleteRow( const unsigned _r ) {
    // Test error.
    if ( _r >= numOfRows() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::deleteRow: "
	  << "Row out of range. "
	  << "It should be <" << numOfRows() << '\0' << std::endl;
      throw out_of_range( msg.str() );
    }    
    // Delete row.
    std::vector< std::vector<T> >::iterator ite = this->begin()+_r;
    this->erase( ite );
  };
  
  /** Inserts a column, moving the rest to the right. 
   * If _c = numOfCols(), it insert it at the end.
   * Obviously, changes number of cols. 
   @param _r Position where the new column will be inserted.
   @param _val Vector containing the new values to be inserted.
   @std::exception invalid_argument if _val has not numOfRows() components.
  */
  virtual void insertCol( const unsigned _c, 
			  const std::vector<T>& _val ) {
    // Test errors.
    if ( _c > numOfCols() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::insertCol: "
	  << "Column out of range. "
	  << "It should be >=" << numOfCols() << '\0' << std::endl;
      throw out_of_range( msg.str() );
    }    
    if ( _val.size() != numOfRows() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::insertCol: "
	  << "Incorrect number of values to be added. "
	  << "It should be ==" << numOfRows() << '\0' << std::endl;
      throw invalid_argument( msg.str() );
    }
    
    // Insert column.
    for( unsigned r=0; r<numOfRows(); ++r ) {
      std::vector<std::vector<T> >::iterator it1 = begin()+r;
      std::vector<T>::iterator it2 = (*it1).begin()+_c;
      (*it1).insert( it2, _val[r] );
    };
  }
  
  /** Eliminates the column at position _c; all the other columns will
      be shifted left.
      @param _c Number of he column to be deleted.
      @std::exception out_of_range if _c >=numOfCols()
  */
  virtual void deleteCol( const unsigned _c ) {
    // Test error.
    if ( _c >= numOfCols() ) {
      std::ostrstream msg;
      msg << "ERROR in eo2dVector::deleteCol: "
	  << "Column out of range. "
	  << "It should be <" << numOfCols() << '\0' << std::endl;
      throw out_of_range( msg.str() );
    }    
    // Delete columns.
    for( unsigned r=0; r<numOfRows(); ++r ) {
      std::vector<std::vector<T> >::iterator it1 = begin()+r;
      std::vector<T>::iterator it2 = (*it1).begin()+_c;
      (*it1).erase( it2 );
    }
  };
  
  /// Returns the number of rows in the eo2d
  virtual unsigned numOfRows() const {
    return size();
  };
  
  /// Returns the number of columns in the eo2d
  virtual unsigned numOfCols() const {
    return begin()->size();
  };
  
  
  /** @name Methods from eoObject
      readFrom and printOn are directly inherited from eo1d
  */
  //@{
  /** Inherited from eoObject 
      @see eoObject
  */
  std::string className() const {return "eo2dVector";};
  //@}
  
};


//____________________________ Some method implementation ____________________

// Ctors_______________________________________________________________________
//_____________________________________________________________________________
template <class T, class fitnessT>
eo2dVector<T,fitnessT>::eo2dVector( const unsigned _rows, 
				    const unsigned _cols, 
				    eoRnd<T>& _rnd )
  : eo2d<T, fitnessT>(), std::vector< std::vector<T> >( _rows, std::vector<T>( _cols, T() ) ){ 
  for ( std::vector< std::vector<T> >::iterator i = begin(); i != end(); ++i ) {
    for( std::vector<T>::iterator j=(*i).begin(); j!= (*i).end(); ++j ) {
      *j = _rnd();
    }
  }
};

//_____________________________________________________________________________
/*template <class T, class fitnessT>
eoVector<T,fitnessT>::eoVector( std::istream& _is)
  : eo1d<T, fitnessT>(), std::vector<T>( ){ 
  while (_is ) {
    T tmp;
    _is >> tmp;
    push_back( tmp );
  }

};
*/

//_____________________________________________________________________________
template <class T, class fitnessT>
std::ostream& operator<<( std::ostream& _os, const eo2dVector<T,fitnessT>& _eo) {
  for( unsigned i=0; i<_eo.numOfRows(); ++i ) {
    for( unsigned j=0; j<_eo.numOfCols(); ++j ) {
      _os << _eo.getGene( i,j ) << " ";
    }
    _os << std::endl;
  }
  return _os;
};

#endif

