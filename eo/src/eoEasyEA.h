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
#include <eoMergeReduce.h>
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
         selectTransform(dummySelect, dummyTransform),
         breed(_breed),
         mergeReduce(dummyMerge, dummyReduce),
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
         selectTransform(dummySelect, dummyTransform),
         breed(_breed),
         mergeReduce(_merge, _reduce),
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
         selectTransform(_select, _transform),
         breed(selectTransform),
         mergeReduce(dummyMerge, dummyReduce),
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
         selectTransform(_select, _transform),
         breed(selectTransform),
         mergeReduce(_merge, _reduce),
         replace(mergeReduce)
         {}




  /// Apply a few generation of evolution to the population.
  virtual void operator()(eoPop<EOT>& _pop)
  {
    eoPop<EOT> offspring;

    do
    {
      try
      {
	       unsigned pSize = _pop.size();
         offspring.clear(); // new offspring

         breed(_pop, offspring);

         apply<EOT>(eval, offspring);

         replace(_pop, offspring); // after replace, the new pop. is in _pop

         if (pSize > _pop.size())
             throw runtime_error("Population shrinking!");
         else if (pSize < _pop.size())
             throw runtime_error("Population growing!");

      }
      catch (exception& e)
      {
	    string s = e.what();
	    s.append( " in eoEasyEA");
	    throw runtime_error( s );
      }
    } while ( continuator( _pop ) );
  }

 private:

  // If selectTransform needs not be used, dummySelect and dummyTransform are used
  // to instantiate it.
     class eoDummySelect : public eoSelect<EOT>
     { public : void operator()(const eoPop<EOT>&, eoPop<EOT>&) {} } dummySelect;

     class eoDummyTransform : public eoTransform<EOT>
     { public : void operator()(eoPop<EOT>&) {} } dummyTransform;


  eoContinue<EOT>&          continuator;
  eoEvalFunc<EOT>&          eval;
  
  eoSelectTransform<EOT>    selectTransform;
  eoBreed<EOT>&             breed;
  
  // If mergeReduce needs not be used, dummyMerge and dummyReduce are used
  // to instantiate it.
  eoNoElitism<EOT>          dummyMerge;
  eoTruncate<EOT>           dummyReduce;

  eoMergeReduce<EOT>        mergeReduce;
  eoReplacement<EOT>&       replace;
};

//-----------------------------------------------------------------------------

#endif eoSelectTransformReduce_h

