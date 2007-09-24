// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoEasyEA.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
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
        } while (continuator(_pop));
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
    { public: /** the dummy functor */
    	void operator()(MOEOT &) {}} dummyEval;
    /** a dummy select */
    class eoDummySelect : public eoSelect < MOEOT >
    { public: /** the dummy functor */
    	void operator()(const eoPop < MOEOT > &, eoPop < MOEOT > &) {} } dummySelect;
    /** a dummy transform */
    class eoDummyTransform : public eoTransform < MOEOT >
    { public: /** the dummy functor */
    	void operator()(eoPop < MOEOT > &) {} } dummyTransform;
    /** a dummy merge */
    eoNoElitism < MOEOT > dummyMerge;
    /** a dummy reduce */
    eoTruncate < MOEOT > dummyReduce;

};

#endif /*MOEOEASYEA_H_*/
