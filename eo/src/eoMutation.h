// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMutation.h
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
#ifndef _EOMUTATION_H
#define _EOMUTATION_H

#include <math.h>
// EO includes
#include <eoOp.h>
#include <eoUniform.h>

/** Generic Mutation of an EO. 
    This is a virtual class, just to establish the interface
*/

template <class EOT>
class eoMutation: public eoMonOp<EOT> {
public:
  
  ///
  eoMutation(const double _rate=0.0) : eoMonOp< EOT >(), rate(_rate) {};
  
  ///
  virtual ~eoMutation() {};
  
  ///
  virtual void operator()( EOT& _eo ) const {
    for ( unsigned i = 0; i < _eo.length(); i ++ )
      applyAt( _eo, i );
  }
  
  /// To print me on a stream.
  /// @param os The ostream.
  void printOn(ostream& os) const {
    os << rate ;
  }
  
  /// To read me from a stream.
  /// @param is The istream.
  void readFrom(istream& is) {
    is >> rate ;
  }
      
  /** @name Methods from eoObject
  */  
  //@{
  /** Inherited from eoObject 
      @see eoObject
  */
  string className() const {return "eoMutation";};
  //@}

protected:
  double rate;  
  
private:  
#ifdef _MSC_VER
  typedef EOT::Type Type;
#else
  typedef typename EOT::Type Type;
#endif
  
  /// applies operator to one gene in the EO. It is empty, so each descent class must define it.
  virtual void applyAt( EOT& _eo, unsigned _i ) const = 0 ;
};



/** Mutation of an eoString.
    The eoString's genes are changed by adding or substracting 1 to 
*/

template <class EOT>
class eoStringMutation: public eoMutation<EOT> {
 public:
  
  ///
  eoStringMutation(const double _rate=0.0) : eoMutation< EOT >(_rate) {};
  
  ///
  virtual ~eoStringMutation() {};
  
  /** @name Methods from eoObject
  */
  //@{
  /** Inherited from eoObject 
      @see eoObject
  */
  string className() const {return "eoStringMutation";};
  //@}
  
  
 private:
  
#ifdef _MSC_VER
  typedef EOT::Type Type;
#else
  typedef typename EOT::Type Type;
#endif
  
  /// applies operator to one gene in the EO. It increments or decrements the value of that gene by one.
  virtual void applyAt( EOT& _eo, unsigned _i ) const {
    eoUniform<double> uniform( 0, 1 );
    if( rate < uniform() ) {
      _eo.gene(_i) += ( uniform()>=0.5 )? (1) : (-1) ;
    }
  }

};

//-----------------------------------------------------------------------------

#endif
