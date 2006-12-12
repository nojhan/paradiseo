/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eo1dWDistance.h 
       Serial EO with distances. Acts as a wrapper for normal eo1ds

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

#ifndef _EO1DWDISTANCE_H
#define _EO1DWDISTANCE_H

#include <iostream>				// for std::ostream

// EO Includes
#include <eo1d.h>
#include <eoDistance.h>

using namespace std;

/** eo1dWDistance: wraps around eo1ds and adds the possibility of computing distances
around them.
*/
template<class T, class fitnessT = float>
class eo1dWDistance: 
  public eo1d< T,fitnessT >, 
  public eoDistance<eo1d<T,fitnessT> > {
public:

  /** Can be used as default ctor; should be called from derived
      classes. Fitness should be  at birth
  */
  eo1dWDistance( eo1d<T,fitnessT>& _eo)
    :eo1d<T,fitnessT> (), eoDistance< eo1d<T,fitnessT> >(), innereo1d( _eo ) {};

  /// Needed virtual dtor
  virtual ~eo1dWDistance(){};

  /** Reads and returns a copy of the gene in position _i
      This implies that T must have a copy ctor .
      @param _i index of the gene, which is the minimal unit. Must be
      an unsigned less than #length()#  
      @return what's inside the gene, with the correct type
	  @std::exception out_of_range if _i > size()
    */
  virtual T getGene( unsigned _i ) const {
    return innereo1d.getGene( _i );
  };

  /** Overwrites the gene placed in position _i with a
   * new value. This means that the assignment operator
   * for T must be defined .
   @param _i index
   @return what's inside the gene, with the correct type
   @std::exception out_of_range if _i > size()
  */
  virtual void setGene( unsigned _i, const T& _value ) {
    innereo1d.setGene( _i, _value);
  };

  /** Inserts a gene, moving the rest to the right. If
   * _i = length(), it should insert it at the end.
   * Obviously, changes length 
   @param _i index
   @param _val new value
  */
  virtual void insertGene( unsigned _i, T _val )  {
    innereo1d.insertGene( _i, _val);
  }

  /** Eliminates the gene at position _i; all the other genes will
      be shifted left
      @param _i index of the gene that is going to be modified.
  */
  virtual void deleteGene( unsigned _i ) {
    innereo1d.deleteGene( _i );
  }

  /// Returns the number of genes in the eo1d
  virtual unsigned length() const {
    return innereo1d.length();
  }

  /// Returns the distance from this EO to the other
  virtual double distance( const eo1d<T,fitnessT>& _eo ) const {
    double tmp = 0;
    // Which one is shorter
    unsigned len = (innereo1d.length() < _eo.length()) ? _eo.length():innereo1d.length();

    // Compute over the whole length. If it does not exists, it counts as 0
    for ( unsigned i = 0; i < len; i ++ ) {
      T val1, val2;
      val1 = ( i > innereo1d.length())?0:innereo1d.getGene(i);
      val2 = ( i > _eo.length() )?0:_eo.getGene(i);
      double diff = val1 - val2;
      tmp += diff*diff;
    }
    return tmp;
  }

  /** Inherited from eoObject 
      @see eoObject
  */
  std::string className() const {return "eo1dWDistance";};

  //@}

private:
  ///Private ctor. Do not use
  eo1dWDistance() {};

  
  eo1d<T,fitnessT>& innereo1d;
};

//@}


#endif
