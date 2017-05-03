/*
  <moeoExtendedESPEA2.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------

#ifndef MOEOEXTENDEDNSGAII_H_
#define MOEOEXTENDEDNSGAII_H_

#include <eoBreed.h>
#include <eoCloneOps.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoGenContinue.h>
#include <eoGeneralBreeder.h>
#include <eoGenOp.h>
#include <algo/moeoEA.h>
#include <replacement/moeoElitistReplacement.h>
#include <selection/moeoDetTournamentSelect.h>
#include <fitness/moeoDominanceDepthFitnessAssignment.h>

#include <diversity/moeoFuzzyCrowdingDiversity.h>


/**
 * Extended NSGAII is an extension of classical algorithm NSGAII for incorporating the aspect of fuzziness in diversity assignement
 */
template < class MOEOT >
class moeoExtendedNSGAII: public moeoEA < MOEOT >
{
public:


    /**
     * Ctor with a eoContinue, a eoPopEval and a eoGenOp.
     * @param _continuator stopping criteria
     * @param _popEval population evaluation function
     * @param _op variation operators
     */
    moeoExtendedNSGAII (eoContinue < MOEOT > & _continuator, eoPopEvalFunc < MOEOT > & _popEval, eoGenOp < MOEOT > & _op) :
            defaultGenContinuator(0), continuator(_continuator), eval(defaultEval), defaultPopEval(eval), popEval(_popEval), select(2),
            selectMany(select,0.0), selectTransform(defaultSelect, defaultTransform), defaultSGAGenOp(defaultQuadOp, 1.0, defaultMonOp, 1.0),
            enBreed(select, _op), breed(genBreed), replace (fitnessAssignment, diversityAssignment)
    {}



    /**
     * Apply a the algorithm to the population _pop until the stopping criteria is satified.
     * @param _pop the population
     */
    virtual void operator () (eoPop < MOEOT > &_pop)
    {
        eoPop < MOEOT > offspring, empty_pop;
        popEval (empty_pop, _pop);	// a first eval of _pop
        // evaluate fitness and diversity
        fitnessAssignment(_pop);
        diversityAssignment(_pop);
        do
        {
            // generate offspring, worths are recalculated if necessary
            breed (_pop, offspring);
            // eval of offspring
            popEval (_pop, offspring);
            // after replace, the new pop is in _pop. Worths are recalculated if necessary
            replace (_pop, offspring);
        }
        while (continuator (_pop));
    }


protected:

    /** a continuator based on the number of generations (used as default) */
    eoGenContinue < MOEOT > defaultGenContinuator;
    /** stopping criteria */
    eoContinue < MOEOT > & continuator;
    /** default eval */
    class DummyEval : public eoEvalFunc < MOEOT >
    {
    public:
        void operator()(MOEOT &) {}
    }
    defaultEval;
    /** evaluation function */
    eoEvalFunc < MOEOT > & eval;
    /** default popEval */
    eoPopLoopEval < MOEOT > defaultPopEval;
    /** evaluation function used to evaluate the whole population */
    eoPopEvalFunc < MOEOT > & popEval;
    /** default select */
    class DummySelect : public eoSelect < MOEOT >
    {
    public :
        void operator()(const eoPop<MOEOT>&, eoPop<MOEOT>&) {}
    }
    defaultSelect;
    /** binary tournament selection */
    moeoDetTournamentSelect < MOEOT > select;
    /** default select many */
    eoSelectMany < MOEOT >  selectMany;
    /** select transform */
    eoSelectTransform < MOEOT > selectTransform;
    /** a default crossover */
    eoQuadCloneOp < MOEOT > defaultQuadOp;
    /** a default mutation */
    eoMonCloneOp < MOEOT > defaultMonOp;
    /** an object for genetic operators (used as default) */
    eoSGAGenOp < MOEOT > defaultSGAGenOp;
    /** default transform */
    class DummyTransform : public eoTransform < MOEOT >
    {
    public :
        void operator()(eoPop<MOEOT>&) {}
    }
    defaultTransform;
    /** general breeder */
    eoGeneralBreeder < MOEOT > genBreed;
    /** breeder */
    eoBreed < MOEOT > & breed;
    /** fitness assignment*/
    FitnessAssignment < MOEOT > fitnessAssignment;
    /** diversity assignment */
    moeoFuzzyCrowdingDiversity  < MOEOT > diversityAssignment;
    /** elitist replacement */
    ElitistReplacement < MOEOT > replace;

};

#endif /*MOEOEXTENDEDNSGAII_H_*/
