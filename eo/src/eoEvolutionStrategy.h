// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoEvolutionStrategy.h
// (c) Maarten Keijzer 2000, GeNeura Team, 1998
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

#ifndef _eoEvolutionStrategy_h
#define _eoEvolutionStrategy_h

//-----------------------------------------------------------------------------

#include <eoEasyEA.h>
#include <eoInplaceTransform.h>
/** eoEvolutionStrategy:
*/

template<class EOT> 
class eoEvolutionStrategy: public eoAlgo<EOT>
{
 public:
     struct plus_strategy{};
     struct comma_strategy{};

     eoEvolutionStrategy(
         eoContinue<EOT>& _continuator,
         eoEvalFunc<EOT>& _eval, 
         eoGOpSelector<EOT>&  _opSel,
         float _lambdaRate,
         comma_strategy)
         :  selectPerc(randomSelect, _lambdaRate),
            transform(_opSel),
            easyEA(_continuator, _eval, selectPerc, transform, noElitism, truncate) 
     {}

     eoEvolutionStrategy(
         eoContinue<EOT>& _continuator,
         eoEvalFunc<EOT>& _eval, 
         eoGOpSelector<EOT>&  _opSel,
         float _lambdaRate,
         plus_strategy)
         :  selectPerc(randomSelect, _lambdaRate),
            transform(_opSel),
            easyEA(_continuator, _eval, selectPerc, transform, plus, truncate) 
     {}


  /// Apply a few generation of evolution to the population.
  virtual void operator()(eoPop<EOT>& _pop) 
  {
      easyEA(_pop);
  }
  
 private:

     eoPlus<EOT>    plus;
     eoNoElitism<EOT> noElitism;
     eoTruncate<EOT> truncate;
     eoSelectRandom<EOT> randomSelect;
     eoSelectPerc<EOT> selectPerc;
     eoInplaceTransform2<EOT> transform;

     /// easyEA is contained rather than a base because of member initialization order!
     eoEasyEA<EOT> easyEA;
};

//-----------------------------------------------------------------------------

#endif eoSelectTransformReduce_h

