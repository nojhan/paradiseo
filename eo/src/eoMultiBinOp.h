// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoMultiBinOp.h
//   Class that combines several binary or unary operators
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

#ifndef _EOMULTIBINOP_h
#define _EOMULTIBINOP_h

#include <iterator>

#include <eoOp.h>

/** MultiMonOp combines several monary operators. By itself, it does nothing to the
EO it´s handled*/
template <class EOT>
class eoMultiBinOp: public eoBinOp<EOT>  {
public:
  /// Ctor from an already existing op
  eoMultiBinOp( const eoBinOp<EOT>* _op )
    : eoBinOp< EOT >( ), vOp(){
    vOp.push_back( _op );
  };

  ///
  eoMultiBinOp( )
    : eoBinOp< EOT >( ), vOp(){};

  /// Ads a new operator
  void adOp( const eoOp<EOT>* _op ){
    vOp.push_back( _op );
  };

  /// needed virtual dtor
  virtual ~eoMultiBinOp() {};
  
  ///
  /// Applies all operators to the EO
  virtual void operator()( EOT& _eo1, EOT& _eo2 ) const {
    if ( vOp.begin() != vOp.end() ) {  // which would mean it's empty
      for ( vector< const eoOp<EOT>* >::const_iterator i = vOp.begin(); 
	    i != vOp.end(); i++ ) {
	// Admits only unary or binary operator
	switch ((*i)->readArity()) {
	case unary:
	  {
	    const eoMonOp<EOT>* monop = static_cast<const eoMonOp<EOT>* >(*i);
	    (*monop)( _eo1 );
	    (*monop)( _eo2 );
	    break;
	  }
	case binary:
	  {
	    const eoBinOp<EOT>* binop = static_cast<const eoBinOp<EOT>* >(*i);
	    (*binop)( _eo1, _eo2 );
	    break;
	  }
	}
      }
    }
  }
  
  
  /** @name Methods from eoObject
      readFrom and printOn are directly inherited from eoOp
  */
  //@{
  /** Inherited from eoObject 
      @see eoObject
  */
  string className() const {return "eoMultiBinOp";};
  //@}

private:

  /// uses pointers to base class since operators can be unary or binary
  vector< const eoOp<EOT>* > vOp;
};

#endif
