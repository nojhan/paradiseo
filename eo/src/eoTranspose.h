// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTranspose.h
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

#ifndef _EOTRANSPOSE_h
#define _EOTRANSPOSE_h

#include <eoUniform.h>

#include <eoOp.h>

/** Transposition operator: interchanges the position of two genes
of an EO. These positions must be defined by an only index, that is,
EOT must subclass eo1d
*/
template <class EOT >
class eoTranspose: public eoMonOp<EOT>  {
public:
  ///
  eoTranspose()
    : eoMonOp< EOT >( ){};
  
  /// needed virtual dtor
  virtual ~eoTranspose() {};
  
  ///
  virtual void operator()( EOT& _eo ) const {
    eoUniform<unsigned> uniform(0, _eo.length() );
    unsigned pos1 = uniform(),
      pos2 = uniform();
    applyAt( _eo, pos1, pos2 );
  }
  
  /** @name Methods from eoObject
      readFrom and printOn are directly inherited from eoOp
  */
  //@{
  /** Inherited from eoObject 
      @see eoObject
  */
  virtual string className() const {return "eoTranspose";};
  //@}
  
private: 
  
#ifdef _MSC_VER
  typedef EOT::Type Type;
#else
  typedef typename EOT::Type Type;
#endif
  
  /** applies operator to one gene in the EO
      @param _eo victim of transposition
      @param i, j positions of the genes that are going to be changed
      @throw runtime_exception if the positions to write are incorrect
  */
  virtual void applyAt( EOT& _eo, unsigned _i, unsigned _j) const {
    try {
      Type tmp = _eo.gene( _i );
      _eo.gene( _i ) =  _eo.gene( _j );
      _eo.gene( _j ) = tmp;
    } catch ( exception& _e ) {
      string msg = _e.what();
      msg += "Caught exception at eoTranspose";
      throw  runtime_error( msg.c_str() );
    }
  }
  
};

#endif
