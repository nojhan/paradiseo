// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGenericMonOp.h
// (c) GeNeura Team, 2000 - EEAAX 1999 - Maarten Keijzer 2000
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
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoGenericMonOp_h
#define _eoGenericMonOp_h

#include <eoOp.h>

/** Contains base classes for generic operators for eoFixedLength 
    and eoVariableLength (They also derive from the eoOp) as well as 
    the corresponding converters to actual Ops.
*/

/** eoGenericMonOp is the generic unary operator:
it takes one argument, and returns a boolean indicating if the argument 
has been modified
*/

template <class EOT>
class eoGenericMonOp: public eoOp<EOT>, public eoUF<EOT&, bool>
{
public:
  /// Ctor
  eoGenericMonOp()
    : eoOp<EOT>( eoOp<EOT>::unary ) {};
  virtual string className() const {return "eoGenericMonOp";};
};

/** COnverter from eoGenericMonOp to eoMonOp 
    the only thinig to do is to transform the boolean into invalidation
*/

template <class EOT>
class eoGeneric2TrueMonOp: public eoMonOp<EOT>
{
public:
  /// Ctor
  eoGeneric2TrueMonOp(eoGenericMonOp<EOT> & _monOp)
    : monOp( _monOp ) {};
  virtual string className() const {return "eoGeneric2trueMonOp";}

  virtual void operator()(EOT & _eo)
    {
      if (monOp(_eo))
	  _eo.invalidate();
    }
  
 private:
  eoGenericMonOp<EOT> & monOp;
};

#endif
