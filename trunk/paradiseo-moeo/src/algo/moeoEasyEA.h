/*
* <moeoEasyEA.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef _MOEOEASYEA_H
#define _MOEOEASYEA_H

#include <apply.h>
#include <eoBreed.h>
#include <eoContinue.h>
#include <eoMergeReduce.h>
#include <eoPopEvalFunc.h>
#include <eoSelect.h>
#include <eoTransform.h>
#include <algo/moeoEA.h>
#include <diversity/moeoDiversityAssignment.h>
#include <diversity/moeoDummyDiversityAssignment.h>
#include <fitness/moeoFitnessAssignment.h>
#include <replacement/moeoReplacement.h>

/**
 * An easy class to design multi-objective evolutionary algorithms.
 */
template < class MOEOT >
class moeoEasyEA: public moeoEA < MOEOT >
  {
  public:

    /**
     * Ctor taking a breed and merge.
     * @param _continuator the stopping criteria
     * @param _eval the evaluation functions
     * @param _breed the breeder
     * @param _replace the replacement strategy
     * @param _fitnessEval the fitness evaluation scheme
     * @param _diversityEval the diversity evaluation scheme
     * @param _evalFitAndDivBeforeSelection put this parameter to 'true' if you want to re-evalue the fitness and the diversity of the population before the selection process
     */
    moeoEasyEA(eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoBreed < MOEOT > & _breed, moeoReplacement < MOEOT > & _replace,
               moeoFitnessAssignment < MOEOT > & _fitnessEval, moeoDiversityAssignment < MOEOT > & _diversityEval, bool _evalFitAndDivBeforeSelection = false)
        :
        continuator(_continuator), eval (_eval), loopEval(_eval), popEval(loopEval), selectTransform(dummySelect, dummyTransform), breed(_breed), mergeReduce(dummyMerge, dummyReduce), replace(_replace),
        fitnessEval(_fitnessEval), diversityEval(_diversityEval), evalFitAndDivBeforeSelection(_evalFitAndDivBeforeSelection)
    {}


    /**
     * Ctor taking a breed, a merge and a eoPopEval.
     * @param _continuator the stopping criteria
     * @param _popEval the evaluation functions for the whole population
     * @param _breed the breeder
     * @param _replace the replacement strategy
     * @param _fitnessEval the fitness evaluation scheme
     * @param _diversityEval the diversity evaluation scheme
     * @param _evalFitAndDivBeforeSelection put this parameter to 'true' if you want to re-evalue the fitness and the diversity of the population before the selection process
     */
    moeoEasyEA(eoContinue < MOEOT > & _continuator, eoPopEvalFunc < MOEOT > & _popEval, eoBreed < MOEOT > & _breed, moeoReplacement < MOEOT > & _replace,
               moeoFitnessAssignment < MOEOT > & _fitnessEval, moeoDiversityAssignment < MOEOT > & _diversityEval, bool _evalFitAndDivBeforeSelection = false)
        :
        continuator(_continuator), eval (dummyEval), loopEval(dummyEval), popEval(_popEval), selectTransform(dummySelect, dummyTransform), breed(_breed), mergeReduce(dummyMerge, dummyReduce), replace(_replace),
        fitnessEval(_fitnessEval), diversityEval(_diversityEval), evalFitAndDivBeforeSelection(_evalFitAndDivBeforeSelection)
    {}


    /**
     * Ctor taking a breed, a merge and a reduce.
     * @param _continuator the stopping criteria
     * @param _eval the evaluation functions
     * @param _breed the breeder
     * @param _merge the merge scheme
     * @param _reduce the reduce scheme
     * @param _fitnessEval the fitness evaluation scheme
     * @param _diversityEval the diversity evaluation scheme
     * @param _evalFitAndDivBeforeSelection put this parameter to 'true' if you want to re-evalue the fitness and the diversity of the population before the selection process
     */
    moeoEasyEA(eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoBreed < MOEOT > & _breed, eoMerge < MOEOT > & _merge, eoReduce< MOEOT > & _reduce,
               moeoFitnessAssignment < MOEOT > & _fitnessEval, moeoDiversityAssignment < MOEOT > & _diversityEval, bool _evalFitAndDivBeforeSelection = false)
        :
        continuator(_continuator), eval(_eval), loopEval(_eval), popEval(loopEval), selectTransform(dummySelect, dummyTransform), breed(_breed), mergeReduce(_merge,_reduce), replace(mergeReduce),
        fitnessEval(_fitnessEval), diversityEval(_diversityEval), evalFitAndDivBeforeSelection(_evalFitAndDivBeforeSelection)
    {}


    /**
     * Ctor taking a select, a transform and a replacement.
     * @param _continuator the stopping criteria
     * @param _eval the evaluation functions
     * @param _select the selection scheme
     * @param _transform the tranformation scheme
     * @param _replace the replacement strategy
     * @param _fitnessEval the fitness evaluation scheme
     * @param _diversityEval the diversity evaluation scheme
     * @param _evalFitAndDivBeforeSelection put this parameter to 'true' if you want to re-evalue the fitness and the diversity of the population before the selection process
     */
    moeoEasyEA(eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoSelect < MOEOT > & _select, eoTransform < MOEOT > & _transform, moeoReplacement < MOEOT > & _replace,
               moeoFitnessAssignment < MOEOT > & _fitnessEval, moeoDiversityAssignment < MOEOT > & _diversityEval, bool _evalFitAndDivBeforeSelection = false)
        :
        continuator(_continuator), eval(_eval), loopEval(_eval), popEval(loopEval), selectTransform(_select, _transform), breed(selectTransform), mergeReduce(dummyMerge, dummyReduce), replace(_replace),
        fitnessEval(_fitnessEval), diversityEval(_diversityEval), evalFitAndDivBeforeSelection(_evalFitAndDivBeforeSelection)
    {}


    /**
     * Ctor taking a select, a transform, a merge and a reduce.
     * @param _continuator the stopping criteria
     * @param _eval the evaluation functions
     * @param _select the selection scheme
     * @param _transform the tranformation scheme
     * @param _merge the merge scheme
     * @param _reduce the reduce scheme
     * @param _fitnessEval the fitness evaluation scheme
     * @param _diversityEval the diversity evaluation scheme
     * @param _evalFitAndDivBeforeSelection put this parameter to 'true' if you want to re-evalue the fitness and the diversity of the population before the selection process
     */
    moeoEasyEA(eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoSelect < MOEOT > & _select, eoTransform < MOEOT > & _transform, eoMerge < MOEOT > & _merge, eoReduce< MOEOT > & _reduce,
               moeoFitnessAssignment < MOEOT > & _fitnessEval, moeoDiversityAssignment < MOEOT > & _diversityEval, bool _evalFitAndDivBeforeSelection = false)
        :
        continuator(_continuator), eval(_eval), loopEval(_eval), popEval(loopEval), selectTransform(_select, _transform), breed(selectTransform), mergeReduce(_merge,_reduce), replace(mergeReduce),
        fitnessEval(_fitnessEval), diversityEval(_diversityEval), evalFitAndDivBeforeSelection(_evalFitAndDivBeforeSelection)
    {}


    /**
     * Applies a few generation of evolution to the population _pop.
     * @param _pop the population
     */
    virtual void operator()(eoPop < MOEOT > & _pop)
    {
      eoPop < MOEOT > offspring, empty_pop;
      popEval(empty_pop, _pop); // A first eval of pop.
      bool firstTime = true;
      do
        {
          try
            {
              unsigned int pSize = _pop.size();
              offspring.clear(); // new offspring
              // fitness and diversity assignment (if you want to or if it is the first generation)
              if (evalFitAndDivBeforeSelection || firstTime)
                {
                  firstTime = false;
                  fitnessEval(_pop);
                  diversityEval(_pop);
                }
              breed(_pop, offspring);
              popEval(_pop, offspring); // eval of parents + offspring if necessary
              replace(_pop, offspring); // after replace, the new pop. is in _pop
              if (pSize > _pop.size())
                {
                  throw std::runtime_error("Population shrinking!");
                }
              else if (pSize < _pop.size())
                {
                  throw std::runtime_error("Population growing!");
                }
            }
          catch (std::exception& e)
            {
              std::string s = e.what();
              s.append( " in moeoEasyEA");
              throw std::runtime_error( s );
            }
        }
      while (continuator(_pop));
    }


  protected:

    /** the stopping criteria */
    eoContinue < MOEOT > & continuator;
    /** the evaluation functions */
    eoEvalFunc < MOEOT > & eval;
    /** to evaluate the whole population */
    eoPopLoopEval < MOEOT > loopEval;
    /** to evaluate the whole population */
    eoPopEvalFunc < MOEOT > & popEval;
    /** breed: a select followed by a transform */
    eoSelectTransform < MOEOT > selectTransform;
    /** the breeder */
    eoBreed < MOEOT > & breed;
    /** replacement: a merge followed by a reduce  */
    eoMergeReduce < MOEOT > mergeReduce;
    /** the replacment strategy */
    moeoReplacement < MOEOT > & replace;
    /** the fitness assignment strategy */
    moeoFitnessAssignment < MOEOT > & fitnessEval;
    /** the diversity assignment strategy */
    moeoDiversityAssignment < MOEOT > & diversityEval;
    /** if this parameter is set to 'true', the fitness and the diversity of the whole population will be re-evaluated before the selection process */
    bool evalFitAndDivBeforeSelection;
    /** a dummy eval */
  class eoDummyEval : public eoEvalFunc < MOEOT >
      {
      public: /** the dummy functor */
        void operator()(MOEOT &)
        {}
      }
    dummyEval;
    /** a dummy select */
  class eoDummySelect : public eoSelect < MOEOT >
      {
      public: /** the dummy functor */
        void operator()(const eoPop < MOEOT > &, eoPop < MOEOT > &)
        {}
      }
    dummySelect;
    /** a dummy transform */
  class eoDummyTransform : public eoTransform < MOEOT >
      {
      public: /** the dummy functor */
        void operator()(eoPop < MOEOT > &)
        {}
      }
    dummyTransform;
    /** a dummy merge */
    eoNoElitism < MOEOT > dummyMerge;
    /** a dummy reduce */
    eoTruncate < MOEOT > dummyReduce;

  };

#endif /*MOEOEASYEA_H_*/
