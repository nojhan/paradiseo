/*
* <moeoNSGA.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
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

#ifndef MOEONSGA_H_
#define MOEONSGA_H_

#include "../../eo/eoBreed.h"
#include "../../eo/eoContinue.h"
#include "../../eo/eoEvalFunc.h"
#include "../../eo/eoGenContinue.h"
#include "../../eo/eoGeneralBreeder.h"
#include "../../eo/eoGenOp.h"
#include "../../eo/eoPopEvalFunc.h"
#include "../../eo/eoSGAGenOp.h"
#include "moeoEA.h"
#include "../diversity/moeoFrontByFrontSharingDiversityAssignment.h"
#include "../fitness/moeoDominanceDepthFitnessAssignment.h"
#include "../replacement/moeoElitistReplacement.h"
#include "../selection/moeoDetTournamentSelect.h"

/**
 * NSGA (Non-dominated Sorting Genetic Algorithm).
 * N. Srinivas, K. Deb, "Multiobjective Optimization Using Nondominated Sorting in Genetic Algorithms".
 * Evolutionary Computation, Vol. 2(3), No 2, pp. 221-248 (1994).
 * This class builds the NSGA algorithm only by using the fine-grained components of the ParadisEO-MOEO framework.
 */
template < class MOEOT >
class moeoNSGA: public moeoEA < MOEOT >
{
public:

    /**
     * Ctor with a crossover, a mutation and their corresponding rates.
     * @param _maxGen maximum number of generations before stopping
     * @param _eval evaluation function
     * @param _crossover crossover
     * @param _pCross crossover probability
     * @param _mutation mutation
     * @param _pMut mutation probability
     * @param _nicheSize niche size
     */
    moeoNSGA (unsigned int _maxGen, eoEvalFunc < MOEOT > & _eval, eoQuadOp < MOEOT > & _crossover, double _pCross, eoMonOp < MOEOT > & _mutation, double _pMut, double _nicheSize = 0.5) :
            defaultGenContinuator(_maxGen), continuator(defaultGenContinuator), eval(_eval), defaultPopEval(_eval), popEval(defaultPopEval), select (2), selectMany(select,0.0), selectTransform(defaultSelect, defaultTransform), defaultSGAGenOp(_crossover, _pCross, _mutation, _pMut), genBreed (select, defaultSGAGenOp), breed (genBreed), diversityAssignment(_nicheSize), replace (fitnessAssignment, diversityAssignment)
    {}


    /**
     * Ctor with a eoContinue and a eoGenOp.
     * @param _continuator stopping criteria
     * @param _eval evaluation function
     * @param _op variation operators
     * @param _nicheSize niche size
     */
    moeoNSGA (eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoGenOp < MOEOT > & _op, double _nicheSize = 0.5) :
            defaultGenContinuator(0), continuator(_continuator), eval(_eval), defaultPopEval(_eval), popEval(defaultPopEval), select(2),
            selectMany(select,0.0), selectTransform(defaultSelect, defaultTransform), defaultSGAGenOp(defaultQuadOp, 1.0, defaultMonOp, 1.0), genBreed(select, _op), breed(genBreed), diversityAssignment(_nicheSize), replace (fitnessAssignment, diversityAssignment)
    {}


    /**
     * Ctor with a eoContinue, a eoPopEval and a eoGenOp.
     * @param _continuator stopping criteria
     * @param _popEval population evaluation function
     * @param _op variation operators
     * @param _nicheSize niche size
     */
    moeoNSGA (eoContinue < MOEOT > & _continuator, eoPopEvalFunc < MOEOT > & _popEval, eoGenOp < MOEOT > & _op, double _nicheSize = 0.5) :
            defaultGenContinuator(0), continuator(_continuator), eval(defaultEval), defaultPopEval(eval), popEval(_popEval), select(2),
            selectMany(select,0.0), selectTransform(defaultSelect, defaultTransform), defaultSGAGenOp(defaultQuadOp, 1.0, defaultMonOp, 1.0), genBreed(select, _op), breed(genBreed), diversityAssignment(_nicheSize), replace (fitnessAssignment, diversityAssignment)
    {}


    /**
     * Ctor with a eoContinue and a eoTransform.
     * @param _continuator stopping criteria
     * @param _eval evaluation function
     * @param _transform variation operator
     * @param _nicheSize niche size
     */
    moeoNSGA (eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoTransform < MOEOT > & _transform, double _nicheSize = 0.5) :
            defaultGenContinuator(0), continuator(_continuator), eval(_eval), defaultPopEval(_eval), popEval(defaultPopEval),
            select(2),  selectMany(select, 1.0), selectTransform(selectMany, _transform), defaultSGAGenOp(defaultQuadOp, 0.0, defaultMonOp, 0.0), genBreed(select, defaultSGAGenOp), breed(selectTransform), diversityAssignment(_nicheSize), replace(fitnessAssignment, diversityAssignment)
    {}


    /**
     * Ctor with a eoContinue, a eoPopEval and a eoTransform.
     * @param _continuator stopping criteria
     * @param _popEval population evaluation function
     * @param _transform variation operator
     * @param _nicheSize niche size
     */
    moeoNSGA (eoContinue < MOEOT > & _continuator, eoPopEvalFunc < MOEOT > & _popEval, eoTransform < MOEOT > & _transform, double _nicheSize = 0.5) :
            defaultGenContinuator(0), continuator(_continuator), eval(defaultEval), defaultPopEval(eval), popEval(_popEval),
            select(2),  selectMany(select, 1.0), selectTransform(selectMany, _transform), defaultSGAGenOp(defaultQuadOp, 0.0, defaultMonOp, 0.0), genBreed(select, defaultSGAGenOp), breed(selectTransform), diversityAssignment(_nicheSize), replace(fitnessAssignment, diversityAssignment)
    {}


    /**
     * Apply the algorithm to the population _pop until the stopping criteria is satified.
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
    /** fitness assignment used in NSGA-II */
    moeoDominanceDepthFitnessAssignment < MOEOT > fitnessAssignment;
    /** diversity assignment used in NSGA-II */
    moeoFrontByFrontSharingDiversityAssignment  < MOEOT > diversityAssignment;
    /** elitist replacement */
    moeoElitistReplacement < MOEOT > replace;

};

#endif /*MOEONSGA_H_*/
