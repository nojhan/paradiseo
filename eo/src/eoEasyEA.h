// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEasyEA.h
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

#ifndef _eoEasyEA_h
#define _eoEasyEA_h

//-----------------------------------------------------------------------------

#include <eoGeneration.h>     // eoPop
#include <eoTerm.h>

/** EOEasyEA:
    An easy-to-use evolutionary algorithm; you can use any chromosome,
    and any selection transformation, merging and evaluation
    algorithms; you can even change in runtime parameters of those
    sub-algorithms 
*/

template<class Chrom> class eoEasyEA: public eoAlgo<Chrom>
{
 public:
  /// Constructor.
  eoEasyEA(eoBinPopOp<Chrom>&    _select, 
	   eoMonPopOp<Chrom>& _transform, 
	   eoBinPopOp<Chrom>&     _replace,
	   eoEvalFunc<Chrom>& _evaluator,
	   eoTerm<Chrom>&     _terminator)
    :step(_select, _transform, _replace, _evaluator), 
    terminator( _terminator){};

  /// Constructor from an already created generation
  eoEasyEA(eoGeneration<Chrom>& _gen,
	   eoTerm<Chrom>&     _terminator):
    step(_gen), 
    terminator( _terminator){};
  
  /// Apply one generation of evolution to the population.
  virtual void operator()(eoPop<Chrom>& pop) {
    while ( terminator( pop ) ) {
      try
	{
	  step(pop);
	}
      catch (exception& e)
	{
	  string s = e.what();
	  s.append( " in eoEasyEA ");
	  throw runtime_error( s );
	}
    } // while
  }
  
  /// Class name.
  string className() const { return "eoEasyEA"; }
  
 private:
  eoGeneration<Chrom>  step;
  eoTerm<Chrom>& terminator;
};

//-----------------------------------------------------------------------------

#endif eoEasyEA_h
