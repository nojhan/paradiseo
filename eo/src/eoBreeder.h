// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoBreeder.h
//   Takes two populations and mixes them
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

#ifndef eoBreeder_h
#define eoBreeder_h

//-----------------------------------------------------------------------------

#include <vector>          // vector
#include <eoUniform.h>     // eoUniform
#include <eoOp.h>          // eoOp, eoMonOp, eoBinOp
#include <eoPop.h>         // eoPop
#include <eoPopOps.h>      // eoTransform
#include <eoOpSelector.h>  // eoOpSelector

#include "eoRandomIndiSelector.h"
#include "eoBackInserter.h"

using namespace std;

/*****************************************************************************
 * eoBreeder: transforms a population using genetic operators.               *
 * For every operator there is a rated to be applyed.                        *
 *****************************************************************************/

template<class Chrom> class eoBreeder: public eoMonPopOp<Chrom>
{
 public:
  /// Default constructor.
  eoBreeder( eoOpSelector<Chrom>& _opSel): opSel( _opSel ) {}
  
  /// Destructor.
  virtual ~eoBreeder() {}

  /**
   * Transforms a population.
   * @param pop The population to be transformed.
   */
  void operator()(eoPop<Chrom>& pop) 
    {
      size_t orgsize = pop.size();

      for (unsigned i = 0; i < pop.size(); i++) {
	eoOp<Chrom>* op = opSel.Op();
	switch (op->getType()) {
    case eoOp<Chrom>::unary:
	  {
	    eoMonOp<Chrom>* monop = static_cast<eoMonOp<Chrom>* >(op);
	    (*monop)( pop[i] );
	    break;
	  }
	case eoOp<Chrom>::binary:
	  {
	    eoBinOp<Chrom>* binop = static_cast<eoBinOp<Chrom>* >(op);
	    eoUniform<unsigned> u(0, pop.size() );
	    (*binop)(pop[i], pop[ u() ] );
	    break;
	  }
	case eoOp<Chrom>::quadratic:
	  {
	    eoQuadraticOp<Chrom>* Qop = static_cast<eoQuadraticOp<Chrom>* >(op);
	    
        eoUniform<unsigned> u(0, pop.size() );
	    (*Qop)(pop[i], pop[ u() ] );
	    break;
      }
    case eoOp<Chrom>::general :
      {
        eoGeneralOp<Chrom>* Gop = static_cast<eoGeneralOp<Chrom>* >(op);

        eoRandomIndiSelector<Chrom> selector;
        eoBackInserter<Chrom>   inserter;

        (*Gop)(selector.init(pop, orgsize, i), inserter.bind(pop));
	    break;
	  }
	}
      }
    };
  
  /// The class name.
  string className() const { return "eoBreeder"; }
  
 private:
  eoOpSelector<Chrom>& opSel;
  
};

//-----------------------------------------------------------------------------

#endif eoBreeder_h

