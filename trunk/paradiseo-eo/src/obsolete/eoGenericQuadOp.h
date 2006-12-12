// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGenericQuadOp.h
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

#ifndef _eoGenericQuadOp_h
#define _eoGenericQuadOp_h

#include <eoOp.h>

/** Contains base classes for generic quadratic operators for eoFixedLength 
    and eoVariableLength (They also derive from the eoOp) as well as 
    the corresponding converters to actual Ops.
*/

/** eoGenericQuadOp is the generic quadratic operator:
it takes two arguments, modifies the first one, and returns a boolean 
indicating if the arguments have actually been modified

WARNING: even if only 1 argument is modified, it should return true, 
         and both fitnesses will be invalidated. It is assumed that 
	 quadratic operators do some exchange of genetic material, so 
	 if one is modified, the other is, too!
*/

template <class EOT>
class eoGenericQuadOp: public eoOp<EOT>, public eoBF<EOT &, EOT &, bool>
{
public:
  /// Ctor
  eoGenericQuadOp()
    : eoOp<EOT>( eoOp<EOT>::quadratic ) {};
  virtual std::string className() const {return "eoGenericQuadOp";};
};

/** Converter from eoGenericQuadOp to eoQuadOp 
    the only thinig to do is to transform the boolean into invalidation
*/

template <class EOT>
class eoGeneric2TrueQuadOp: public eoQuadOp<EOT>
{
public:
  /// Ctor
  eoGeneric2TrueQuadOp(eoGenericQuadOp<EOT> & _quadOp)
    : quadOp( _quadOp ) {};
  virtual std::string className() const {return "eoGeneric2TrueQuadOp";}

  virtual void operator()(EOT & _eo1, EOT & _eo2)
    {
      if (quadOp(_eo1, _eo2))
	{
	  _eo1.invalidate();
	  _eo2.invalidate();
	}
    }
  
 private:
  eoGenericQuadOp<EOT> & quadOp;
};

#endif
