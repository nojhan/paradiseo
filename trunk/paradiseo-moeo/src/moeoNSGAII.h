// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoNSGAII.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEONSGAII_H_
#define MOEONSGAII_H_

#include <eoGeneralBreeder.h>
#include <eoBreed.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoGenContinue.h>
#include <eoGenOp.h>
#include <eoPopEvalFunc.h>
#include <eoSGAGenOp.h>
#include <moeoCrowdingDistanceDiversityAssignment.h>
#include <moeoDetTournamentSelect.h>
#include <moeoElitistReplacement.h>
#include <moeoFastNonDominatedSortingFitnessAssignment.h>

/**
 * The NSGA-II algorithm as described in:
 * Deb, K., S. Agrawal, A. Pratap, and T. Meyarivan : "A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II".
 * In IEEE Transactions on Evolutionary Computation, Vol. 6, No 2, pp 182-197 (April 2002).
 * This class builds the NSGA-II algorithm only by using the components of the ParadisEO-MOEO framework.
 */
template < class MOEOT >
class moeoNSGAII: public moeoEA < MOEOT >
{
public:

    /**
     * This constructor builds the algorithm as descibed in the paper.
     * @param _max_gen number of generations before stopping
     * @param _eval evaluation function
     * @param _op variation operator
    */
    moeoNSGAII (unsigned _max_gen, eoEvalFunc < MOEOT > & _eval, eoGenOp < MOEOT > &_op) :
            continuator (*(new eoGenContinue < MOEOT > (_max_gen))), eval (_eval), loopEval (_eval), popEval (loopEval), select (2),	// binary tournament selection
            replace (fitnessAssignment, diversityAssignment), genBreed (select, _op), breed (genBreed)
    {}


    /**
     * Ctor taking _max_gen, crossover and mutation.
     * @param _max_gen number of generations before stopping
    * @param _eval evaluation function
    * @param _crossover crossover
    * @param _pCross crossover probability
    * @param _mutation mutation
    * @param _pMut mutation probability
     */
    moeoNSGAII (unsigned _max_gen, eoEvalFunc < MOEOT > &_eval, eoQuadOp < MOEOT > & _crossover, double _pCross, eoMonOp < MOEOT > & _mutation, double _pMut) :
            continuator (*(new eoGenContinue < MOEOT > (_max_gen))), eval (_eval), loopEval (_eval), popEval (loopEval), select (2),	// binary tournament selection
            replace (fitnessAssignment, diversityAssignment), genBreed (select, *new eoSGAGenOp < MOEOT > (_crossover, _pCross, _mutation, _pMut)), breed (genBreed)
    {}


    /**
     * Ctor taking a continuator instead of _gen_max.
     * @param _continuator stopping criteria
     * @param _eval evaluation function
     * @param _op variation operator
     */
    moeoNSGAII (eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoGenOp < MOEOT > & _op) :
            continuator (_continuator), eval (_eval), loopEval (_eval), popEval (loopEval), select (2),	// binary tournament selection
            replace (fitnessAssignment, diversityAssignment), genBreed (select, _op), breed (genBreed)
    {}


    /**
     * Apply a few generation of evolution to the population _pop.
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
        } while (continuator (_pop));
    }


protected:

    /** stopping criteria */
    eoContinue < MOEOT > & continuator;
    /** evaluation function */
    eoEvalFunc < MOEOT > & eval;
    /** to evaluate the whole population */
    eoPopLoopEval < MOEOT > loopEval;
    /** to evaluate the whole population */
    eoPopEvalFunc < MOEOT > & popEval;
    /** binary tournament selection */
    moeoDetTournamentSelect < MOEOT > select;
    /** elitist replacement */
    moeoElitistReplacement < MOEOT > replace;
    /** general breeder */
    eoGeneralBreeder < MOEOT > genBreed;
    /** breeder */
    eoBreed < MOEOT > & breed;
    /** fitness assignment used in NSGA-II */
    moeoFastNonDominatedSortingFitnessAssignment < MOEOT > fitnessAssignment;
    /** Diversity assignment used in NSGA-II */
    moeoCrowdingDistanceDiversityAssignment  < MOEOT > diversityAssignment;

};

#endif /*MOEONSGAII_H_*/
