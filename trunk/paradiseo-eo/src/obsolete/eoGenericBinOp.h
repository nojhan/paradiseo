// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGenericBinOp.h
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

#ifndef _eoGenericBinOp_h
#define _eoGenericBinOp_h

#include <eoOp.h>

/** Contains base classes for generic binary operators for eoFixedLength 
    and eoVariableLength (They also derive from the eoOp) as well as 
    the corresponding converters to actual Ops.
*/

/** eoGenericBinOp is the generic binary operator:
it takes two arguments, modifies the first one, and returns a boolean 
indicating if the argument has actually been modified
*/

template <class EOT>
class eoGenericBinOp: public eoOp<EOT>, public eoBF<EOT &, const EOT &, bool>
{
public:
  /// Ctor
  eoGenericBinOp()
    : eoOp<EOT>( eoOp<EOT>::binary ) {};
  virtual std::string className() const {return "eoGenericBinOp";};
};

/** Converter from eoGenericBinOp to eoBinOp 
    the only thinig to do is to transform the boolean into invalidation
*/

template <class EOT>
class eoGeneric2TrueBinOp: public eoBinOp<EOT>
{
public:
  /// Ctor
  eoGeneric2TrueBinOp(eoGenericBinOp<EOT> & _binOp)
    : binOp( _binOp ) {};
  virtual std::string className() const {return "eoGeneric2TrueBinOp";}

  virtual void operator()(EOT & _eo1, const EOT & _eo2)
    {
      if (binOp(_eo1, _eo2))
	  _eo1.invalidate();
    }
  
 private:
  eoGenericBinOp<EOT> & binOp;
};

#endif
