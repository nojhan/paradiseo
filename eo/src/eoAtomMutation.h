// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoAtomMutation.h
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
#ifndef _EOATOMMUTATION_H
#define _EOATOMMUTATION_H

// STL includes
#include <iterator>

// EO includes
#include <eoOp.h>
#include <eoUniform.h>
#include <eoRNG.h>
#include <eoAtomMutator.h>

/** Atomic mutation of an EO. Acts on containers, and applies a mutation
    operator to each element of the container with some probability. EOT must
    be a container of any type
*/
template <class EOT>
class eoAtomMutation: public eoMonOp<EOT> {
public:

#ifdef _MSC_VER
  typedef EOT::Type Type;
#else
  typedef typename EOT::Type Type;
#endif

  /// 
  eoAtomMutation(const eoAtomMutator<Type>& _atomMut, const double _rate=0.0) 
    : eoMonOp< EOT >(), rate(_rate), atomMutator( _atomMut ) {};
  
  ///
  virtual ~eoAtomMutation() {};
  
  ///
  virtual void operator()( EOT& _eo ) const {
    typename EOT::iterator i;
    for ( i = _eo.begin(); i != _eo.end(); i ++ )
      if ( rng.flip( rate ) ) {
	atomMutator( *i );
      }
  }
  
  /** To print me on a stream.
      @param os The ostream.
  */
  void printOn(ostream& os) const {
    os << rate ;
  }
  
  /** To read me from a stream.
      @param is The istream */
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

private:  

  double rate;  
  const eoAtomMutator<Type>& atomMutator;
};


#endif

