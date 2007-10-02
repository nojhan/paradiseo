/* <moeoNSGA.h>  
 *
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
 */
 
#ifndef MOEONSGA_H_
#define MOEONSGA_H_

#include <eoBreed.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoGenContinue.h>
#include <eoGeneralBreeder.h>
#include <eoGenOp.h>
#include <eoPopEvalFunc.h>
#include <eoSGAGenOp.h>
#include <algo/moeoEA.h>
#include <diversity/moeoFrontByFrontSharingDiversityAssignment.h>
#include <fitness/moeoFastNonDominatedSortingFitnessAssignment.h>
#include <replacement/moeoElitistReplacement.h>
#include <selection/moeoDetTournamentSelect.h>

/**
 * NSGA (Non-dominated Sorting Genetic Algorithm) as described in:
 * N. Srinivas, K. Deb, "Multiobjective Optimization Using Nondominated Sorting in Genetic Algorithms".
 * Evolutionary Computation, Vol. 2(3), No 2, pp. 221-248 (1994).
 * This class builds the NSGA algorithm only by using the fine-grained components of the ParadisEO-MOEO framework.
 */
template < class MOEOT >
class moeoNSGA: public moeoEA < MOEOT >
{
public:

    /**
     * Simple ctor with a eoGenOp.
     * @param _maxGen number of generations before stopping
     * @param _eval evaluation function
     * @param _op variation operator
     * @param _nicheSize niche size
     */
    moeoNSGA (unsigned int _maxGen, eoEvalFunc < MOEOT > & _eval, eoGenOp < MOEOT > & _op, double _nicheSize = 0.5) :
            defaultGenContinuator(_maxGen), continuator(defaultGenContinuator), popEval(_eval), select(2),
            diversityAssignment(_nicheSize), replace(fitnessAssignment, diversityAssignment), genBreed(select, _op), breed(genBreed)
    {}


    /**
     * Simple ctor with a eoTransform.
     * @param _maxGen number of generations before stopping
     * @param _eval evaluation function
     * @param _op variation operator
     * @param _nicheSize niche size
     */
    moeoNSGA (unsigned int _maxGen, eoEvalFunc < MOEOT > & _eval, eoTransform < MOEOT > & _op, double _nicheSize = 0.5) :
            defaultGenContinuator(_maxGen), continuator(defaultGenContinuator), popEval(_eval), select(2),
            diversityAssignment(_nicheSize), replace(fitnessAssignment, diversityAssignment), genBreed(select, _op), breed(genBreed)
    {}


    /**
     * Ctor with a crossover, a mutation and their corresponding rates.
     * @param _maxGen number of generations before stopping
     * @param _eval evaluation function
     * @param _crossover crossover
     * @param _pCross crossover probability
     * @param _mutation mutation
     * @param _pMut mutation probability
     * @param _nicheSize niche size
     */
    moeoNSGA (unsigned int _maxGen, eoEvalFunc < MOEOT > & _eval, eoQuadOp < MOEOT > & _crossover, double _pCross, eoMonOp < MOEOT > & _mutation, double _pMut, double _nicheSize = 0.5) :
            defaultGenContinuator(_maxGen), continuator(defaultGenContinuator), popEval(_eval), select (2),
            diversityAssignment(_nicheSize), replace (fitnessAssignment, diversityAssignment),
            defaultSGAGenOp(_crossover, _pCross, _mutation, _pMut), genBreed (select, defaultSGAGenOp), breed (genBreed)
    {}


    /**
     * Ctor with a continuator (instead of _maxGen) and a eoGenOp.
     * @param _continuator stopping criteria
     * @param _eval evaluation function
     * @param _op variation operator
     * @param _nicheSize niche size
     */
    moeoNSGA (eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoGenOp < MOEOT > & _op, double _nicheSize = 0.5) :
            continuator(_continuator), popEval(_eval), select(2),
            diversityAssignment(_nicheSize), replace(fitnessAssignment, diversityAssignment), genBreed(select, _op), breed(genBreed)
    {}


    /**
     * Ctor with a continuator (instead of _maxGen) and a eoTransform.
     * @param _continuator stopping criteria
     * @param _eval evaluation function
     * @param _op variation operator
     * @param _nicheSize niche size
     */
    moeoNSGA (eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoTransform < MOEOT > & _op, double _nicheSize = 0.5) :
            continuator(_continuator), popEval(_eval), select(2),
            diversityAssignment(_nicheSize), replace(fitnessAssignment, diversityAssignment), genBreed(select, _op), breed(genBreed)
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

    /** a continuator based on the number of generations (used as default) */
    eoGenContinue < MOEOT > defaultGenContinuator;
    /** stopping criteria */
    eoContinue < MOEOT > & continuator;
    /** evaluation function used to evaluate the whole population */
    eoPopLoopEval < MOEOT > popEval;
    /** binary tournament selection */
    moeoDetTournamentSelect < MOEOT > select;
    /** fitness assignment used in NSGA-II */
    moeoFastNonDominatedSortingFitnessAssignment < MOEOT > fitnessAssignment;
    /** diversity assignment used in NSGA-II */
    moeoFrontByFrontSharingDiversityAssignment  < MOEOT > diversityAssignment;
    /** elitist replacement */
    moeoElitistReplacement < MOEOT > replace;
    /** an object for genetic operators (used as default) */
    eoSGAGenOp < MOEOT > defaultSGAGenOp;
    /** general breeder */
    eoGeneralBreeder < MOEOT > genBreed;
    /** breeder */
    eoBreed < MOEOT > & breed;

};

#endif /*MOEONSGAII_H_*/
