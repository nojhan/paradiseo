// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoOp.h
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

#ifndef eoGeneration_h
#define eoGeneration_h

//-----------------------------------------------------------------------------

#include <eoAlgo.h>     // eoPop
#include <eoEvalFunc.h> 
#include <eoPopOps.h>  // eoSelect, eoTranform, eoMerge

//-----------------------------------------------------------------------------

/** eoGeneration
 * Single step of a evolutionary algorithm. Applies selection, then genetic
 * operators, replaces using a replacement policy, and finally evaluates the
 * new ones */
template<class Chrom> class eoGeneration: public eoAlgo<Chrom>
{
 public:
  /// Constructor.
  eoGeneration(eoBinPopOp<Chrom>&    _select, 
	       eoMonPopOp<Chrom>& _transform, 
	       eoBinPopOp<Chrom>&     _replace,
	       eoEvalFunc<Chrom>& _evaluator):
    select(_select), transform(_transform), 
    replace(_replace), evaluator( _evaluator) {};

  /// Copy Constructor.
  eoGeneration(eoGeneration<Chrom>& _gen):
    select(_gen.select), transform(_gen.transform), 
    replace(_gen.replace), evaluator( _gen.evaluator ) {};


  /// Apply one generation of evolution to the population.
  virtual void operator()(eoPop<Chrom>& pop) {
      eoPop<Chrom> breeders;      
      select(pop, breeders);
      transform(breeders);
      eoPop<Chrom>::iterator i;
      // Can't use foreach here since foreach takes the 
      // parameter by reference
      for ( i = breeders.begin(); i != breeders.end(); i++)
	evaluator(*i);
      replace(breeders, pop);
    }
  
  /// Class name.
  string className() const { return "eoGeneration"; }
  
 private:
  eoBinPopOp<Chrom>&    select;
  eoMonPopOp<Chrom>& transform;
  eoBinPopOp<Chrom>&     replace;
  eoEvalFunc<Chrom>& evaluator;
};

//-----------------------------------------------------------------------------

#endif eoGeneration_h
