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

#include <eoAlgo.h>
#include <eoContinue.h>
#include <eoSelect.h>
#include <eoTransform.h>
#include <eoBreed.h>
#include <eoMerge.h>
#include <eoReduce.h>
#include <eoReplacement.h>

/** eoEasyEA:
    An easy-to-use evolutionary algorithm; you can use any chromosome,
    and any selection transformation, merging and evaluation
    algorithms; you can even change in runtime parameters of those
    sub-algorithms 
*/

template<class EOT> class eoEasyEA: public eoAlgo<EOT>
{
 public:

  /// Ctor taking a breed and merge.
     eoEasyEA(
         eoContinue<EOT>& _continuator,
         eoEvalFunc<EOT>& _eval,
         eoBreed<EOT>& _breed,
         eoReplacement<EOT>& _replace     
     ) : continuator(_continuator), 
         eval(_eval),
         selectTransform(0),
         breed(_breed),
         mergeReduce(0),
         replace(_replace)
         {}

  /// Ctor eoBreed, eoMerge and eoReduce.
    eoEasyEA(
         eoContinue<EOT>& _continuator,
         eoEvalFunc<EOT>& _eval,
         eoBreed<EOT>& _breed,
         eoMerge<EOT>& _merge,
         eoReduce<EOT>& _reduce
     ) : continuator(_continuator), 
         eval(_eval),
         selectTransform(0),
         breed(_breed),
         mergeReduce(new eoMergeReduce<EOT>(_merge, _reduce)),
         replace(mergeReduce)
         {}

  /// Ctor eoSelect, eoTransform, and eoReplacement
    eoEasyEA(
         eoContinue<EOT>& _continuator,
         eoEvalFunc<EOT>& _eval,
         eoSelect<EOT>& _select,
         eoTransform<EOT>& _transform,
         eoReplacement<EOT>& _replace     
     ) : continuator(_continuator), 
         eval(_eval),
         selectTransform(new eoSelectTransform<EOT>(_select, _transform)),
         breed(selectTransform),
         mergeReduce(0),
         replace(_replace)
         {}

    /// Ctor eoSelect, eoTransform, eoMerge and eoReduce.
     eoEasyEA(
         eoContinue<EOT>& _continuator,
         eoEvalFunc<EOT>& _eval,
         eoSelect<EOT>& _select,
         eoTransform<EOT>& _transform,
         eoMerge<EOT>&     _merge,
         eoReduce<EOT>&    _reduce
     ) : continuator(_continuator), 
         eval(_eval),
         selectTransform(new eoSelectTransform<EOT>(_select, _transform)),
         breed(*selectTransform),
         mergeReduce(new eoMergeReduce<EOT>(_merge, _reduce)),
         replace(*mergeReduce)
         {}



         
     ~eoEasyEA() { delete selectTransform; delete mergeReduce; }

  /// Apply a few generation of evolution to the population.
  virtual void operator()(eoPop<EOT>& _pop) 
  {
    eoPop<EOT> offspring;
    
    while ( continuator( _pop ) ) 
    {
      try
      {
         breed(_pop, offspring);
         
         apply<EOT>(eval, offspring);
         
         replace(_pop, offspring);

         if (offspring.size() < _pop.size())
             throw runtime_error("Population shrinking!");
         else if (offspring.size() > _pop.size())
             throw runtime_error("Population growing!");

         _pop.swap(offspring);

      }
      catch (exception& e)
      {
	    string s = e.what();
	    s.append( " in eoSelectTransformReduce ");
	    throw runtime_error( s );
      }
    } // while
  }
  
 private:

     /// dissallow copying cuz of pointer stuff
     eoEasyEA(const eoEasyEA&);
     /// dissallow copying cuz of pointer stuff
     const eoEasyEA& operator=(const eoEasyEA&);

  eoContinue<EOT>&          continuator;
  eoEvalFunc<EOT>&          eval;
  eoSelectTransform<EOT>*   selectTransform;
  eoBreed<EOT>&             breed;
  eoMergeReduce<EOT>*       mergeReduce;
  eoReplacement<EOT>&       replace;
};

//-----------------------------------------------------------------------------

#endif eoSelectTransformReduce_h

