// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoIBEA.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOIBEA_H_
#define MOEOIBEA_H_

#include <eoGeneralBreeder.h>
#include <eoBreed.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoGenContinue.h>
#include <eoGenOp.h>
#include <eoPopEvalFunc.h>
#include <eoSGAGenOp.h>
#include <moeoDetTournamentSelect.h>
#include <moeoDiversityAssignment.h>
#include <moeoIndicatorBasedFitnessAssignment.h>
#include <moeoEnvironmentalReplacement.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>

/**
 * IBEA (Indicator-Based Evolutionary Algorithm) as described in:
 * E. Zitzler, S. Künzli, "Indicator-Based Selection in Multiobjective Search", Proc. 8th International Conference on Parallel Problem Solving from Nature (PPSN VIII), pp. 832-842, Birmingham, UK (2004).
 * This class builds the IBEA algorithm only by using the fine-grained components of the ParadisEO-MOEO framework.
 */
template < class MOEOT >
class moeoIBEA : public moeoEA < MOEOT >
{
public:

    /** The type of objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Simple ctor with a eoGenOp.
     * @param _maxGen number of generations before stopping
     * @param _eval evaluation function
     * @param _op variation operator
     */
    moeoIBEA (unsigned _maxGen, eoEvalFunc < MOEOT > & _eval, eoGenOp < MOEOT > & _op, moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > & _metric, const double _kappa=0.05) :
            defaultGenContinuator(_maxGen), continuator(defaultGenContinuator), popEval(_eval), select(2),
            fitnessAssignment(_metric, _kappa), replace(fitnessAssignment, dummyDiversityAssignment), genBreed(select, _op), breed(genBreed)
    {}


    /**
     * Simple ctor with a eoTransform.
     * @param _maxGen number of generations before stopping
     * @param _eval evaluation function
     * @param _op variation operator
     */
    moeoIBEA (unsigned _maxGen, eoEvalFunc < MOEOT > & _eval, eoTransform < MOEOT > & _op, moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > & _metric, const double _kappa=0.05) :
            defaultGenContinuator(_maxGen), continuator(defaultGenContinuator), popEval(_eval), select(2),
            fitnessAssignment(_metric, _kappa), replace(fitnessAssignment, dummyDiversityAssignment), genBreed(select, _op), breed(genBreed)
    {}


    /**
     * Ctor with a crossover, a mutation and their corresponding rates.
     * @param _maxGen number of generations before stopping
     * @param _eval evaluation function
     * @param _crossover crossover
     * @param _pCross crossover probability
     * @param _mutation mutation
     * @param _pMut mutation probability
     */
    moeoIBEA (unsigned _maxGen, eoEvalFunc < MOEOT > & _eval, eoQuadOp < MOEOT > & _crossover, double _pCross, eoMonOp < MOEOT > & _mutation, double _pMut, moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > & _metric, const double _kappa=0.05) :
            defaultGenContinuator(_maxGen), continuator(defaultGenContinuator), popEval(_eval), select (2),
            fitnessAssignment(_metric, _kappa), replace (fitnessAssignment, dummyDiversityAssignment), defaultSGAGenOp(_crossover, _pCross, _mutation, _pMut),
            genBreed (select, defaultSGAGenOp), breed (genBreed)
    {}


    /**
     * Ctor with a continuator (instead of _maxGen) and a eoGenOp.
     * @param _continuator stopping criteria
     * @param _eval evaluation function
     * @param _op variation operator
     */
    moeoIBEA (eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoGenOp < MOEOT > & _op, moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > & _metric, const double _kappa=0.05) :
            continuator(_continuator), popEval(_eval), select(2),
            fitnessAssignment(_metric, _kappa), replace(fitnessAssignment, dummyDiversityAssignment), genBreed(select, _op), breed(genBreed)
    {}


    /**
     * Ctor with a continuator (instead of _maxGen) and a eoTransform.
     * @param _continuator stopping criteria
     * @param _eval evaluation function
     * @param _op variation operator
     */
    moeoIBEA (eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoTransform < MOEOT > & _op, moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > & _metric, const double _kappa=0.05) :
            continuator(_continuator), popEval(_eval), select(2),
            fitnessAssignment(_metric, _kappa), replace(fitnessAssignment, dummyDiversityAssignment), genBreed(select, _op), breed(genBreed)
    {}


    /**
     * Apply a few generation of evolution to the population _pop until the stopping criteria is verified.
     * @param _pop the population
     */
    virtual void operator () (eoPop < MOEOT > &_pop)
    {
        eoPop < MOEOT > offspring, empty_pop;
        popEval (empty_pop, _pop);	// a first eval of _pop
        // evaluate fitness and diversity
        fitnessAssignment(_pop);
        dummyDiversityAssignment(_pop);
        do
        {
            // generate offspring, worths are recalculated if necessary
            breed (_pop, offspring);
            // eval of offspring
            popEval (_pop, offspring);
            // after replace, the new pop is in _pop. Worths are recalculated if necessary
            replace (_pop, offspring);
        } while (continuator (_pop));
    }


protected:

    /** a continuator based on the number of generations (used as default) */
    eoGenContinue < MOEOT > defaultGenContinuator;
    /** stopping criteria */
    eoContinue < MOEOT > & continuator;
    /** evaluation function used to evaluate the whole population */
    eoPopLoopEval < MOEOT > popEval;
    /** binary tournament selection */
    moeoDetTournamentSelect < MOEOT > select;
    /** fitness assignment used in IBEA */
    moeoIndicatorBasedFitnessAssignment < MOEOT > fitnessAssignment;
    /** dummy diversity assignment */
    moeoDummyDiversityAssignment < MOEOT > dummyDiversityAssignment;
    /** elitist replacement */
    moeoEnvironmentalReplacement < MOEOT > replace;
    /** an object for genetic operators (used as default) */
    eoSGAGenOp < MOEOT > defaultSGAGenOp;
    /** general breeder */
    eoGeneralBreeder < MOEOT > genBreed;
    /** breeder */
    eoBreed < MOEOT > & breed;

};

#endif /*MOEOIBEA_H_*/
