// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoInplaceTransform.h
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

#ifndef eoInplaceTransform_h
#define eoInplaceTransform_h

//-----------------------------------------------------------------------------

#include <vector>          // std::vector
#include <utils/eoRNG.h>
#include <eoOp.h>          // eoOp, eoMonOp, eoBinOp
#include <eoPop.h>         // eoPop
#include <eoOpSelector.h>  // eoOpSelector
#include <eoFunctor.h>
#include <eoIndiSelector.h>
#include <eoBackInserter.h>

/*****************************************************************************
 * eoInplaceTransform1: transforms a population using genetic operators.               *
 * It does it in an SGA like manner                                 
 *****************************************************************************/
template<class Chrom> class eoInplaceTransform1 : public eoTransform<Chrom>
{
 public:
    
  /// Default constructor.
  eoInplaceTransform1( eoOpSelector<Chrom>& _opSel): opSel( _opSel ), select(defaultSelect) {}
  eoInplaceTransform1( eoOpSelector<Chrom>& _opSel, eoSelectOne<Chrom>& _select)
      : opSel(_opSel), select(_select) {}
  /**
   * Transforms a population.
   * @param pop The population to be transformed.
   */
  void operator()(eoPop<Chrom>& pop) 
    {
      // copy the guys in a newpop
      // because otherwise eoSelectRandom might select freshly created individuals
      eoPop<Chrom> newpop;
      newpop.reserve(pop.size());

      // Set up general op helper classes
      eoSelectOneIndiSelector<Chrom> inplace(select);
      eoBackInserter<Chrom> inserter;

      // set up selection routine
      select.setup(pop);

      for (unsigned i = 0; i < pop.size(); i++) 
      {
	     eoOp<Chrom>* op = opSel.Op();
	
         switch (op->getType()) 
        {
            case eoOp<Chrom>::unary:
	        {
	        eoMonOp<Chrom>* monop = static_cast<eoMonOp<Chrom>* >(op);
	        newpop.push_back(pop[i]);
            (*monop)( newpop.back() );
	        break;
	    }
	    case eoOp<Chrom>::binary:
	    {
	        eoBinOp<Chrom>* binop = static_cast<eoBinOp<Chrom>* >(op);
	        newpop.push_back(pop[i]);
            (*binop)(newpop[i], select(pop));
	        break;
	    }
	    case eoOp<Chrom>::quadratic:
	    {

	        eoQuadraticOp<Chrom>* Qop = static_cast<eoQuadraticOp<Chrom>* >(op);
	    
            newpop.push_back(pop[i]);
            Chrom& indy1 = newpop.back();
            
            if (++i == pop.size())
                newpop.push_back(select(pop));
            else
                newpop.push_back(pop[i]);

	        (*Qop)(indy1, newpop.back() );
	        break;
        }
        case eoOp<Chrom>::general :
        {
            eoGeneralOp<Chrom>* Gop = static_cast<eoGeneralOp<Chrom>* >(op);
        
            inplace.bind(pop);
            inplace.bias(i,i + 1);

            inserter.bind(newpop);
            unsigned orgsize = newpop.size();
            (*Gop)(inplace, inserter);
            unsigned diff = newpop.size() - orgsize;
            i = i + (diff-1);
	        break;
	    }
	}
         pop.swap(newpop); // overwrite existing pop
      }
    };
    
 private:
  eoOpSelector<Chrom>& opSel;
  eoRandomSelect<Chrom> defaultSelect;
  eoSelectOne<Chrom>&  select;
  
};

#include <eoGOpSelector.h>

/*****************************************************************************
 * eoInplaceTransform2: transforms a population using general genetic operators.               *
 * It does it in an SGA like manner                                 
 *****************************************************************************/
template<class Chrom> class eoInplaceTransform2 : public eoTransform<Chrom>
{
 public:
    
  /// Default constructor.
  eoInplaceTransform2( eoGOpSelector<Chrom>& _opSel): opSel( _opSel ), select(defaultSelect) {}
  eoInplaceTransform2( eoGOpSelector<Chrom>& _opSel, eoSelectOne<Chrom>& _select)
      : opSel(_opSel), select(_select) {}
  /**
   * Transforms a population.
   * @param pop The population to be transformed.
   */
  void operator()(eoPop<Chrom>& pop) 
    {
      // copy the guys in a newpop
      // because otherwise eoSelectRandom might select freshly created individuals
      eoPop<Chrom> newpop;
      newpop.reserve(pop.size());

      // Set up general op helper classes
      eoSelectOneIndiSelector<Chrom> inplace(select);
      eoBackInserter<Chrom> inserter;

      // set up selection routine
      select.setup(pop);

      for (unsigned i = 0; i < pop.size(); i++) 
      {
            eoGeneralOp<Chrom>& Gop = opSel.selectOp();
        
            inplace.bind(pop);
            inplace.bias(i,i + 1);

            inserter.bind(newpop);
            unsigned orgsize = newpop.size();
            Gop(inplace, inserter);
            
            // see how many have been inserted and add that to i (minus one ofcourse)
            unsigned diff = newpop.size() - orgsize;
            i = i + (diff-1);
	   }
       
      pop.swap(newpop); // overwrite existing pop
    }
    
 private:
  eoGOpSelector<Chrom>& opSel;
  eoRandomSelect<Chrom> defaultSelect;
  eoSelectOne<Chrom>&  select;
  
};


//-----------------------------------------------------------------------------

#endif eoBreeder_h

