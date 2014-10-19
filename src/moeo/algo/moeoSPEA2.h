/*
* <moeoSPEA2.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Lille-Nord Europe, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jeremie Humeau
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
// moeoSPEA2.h
//-----------------------------------------------------------------------------
#ifndef MOEOSPEA2_H_
#define MOEOSPEA2_H_

#include "../../eo/eoBreed.h"
#include "../../eo/eoCloneOps.h"
#include "../../eo/eoContinue.h"
#include "../../eo/eoEvalFunc.h"
#include "../../eo/eoGenContinue.h"
#include "../../eo/eoGeneralBreeder.h"
#include "../../eo/eoGenOp.h"
#include "../../eo/eoPopEvalFunc.h"
#include "../../eo/eoSGAGenOp.h"
#include "moeoEA.h"
#include "../diversity/moeoNearestNeighborDiversityAssignment.h"
#include "../fitness/moeoDominanceCountFitnessAssignment.h"
#include "../fitness/moeoDominanceCountRankingFitnessAssignment.h"
#include "../replacement/moeoGenerationalReplacement.h"
#include "../selection/moeoDetTournamentSelect.h"
#include "../archive/moeoFixedSizeArchive.h"
#include "../distance/moeoEuclideanDistance.h"
#include "../selection/moeoSelectFromPopAndArch.h"

/**
 * SPEA2 algorithm.
 * E. Zitzler, M. Laumanns, and L. Thiele. SPEA2: Improving the Strength Pareto Evolutionary Algorithm. Technical Report 103,
 * Computer Engineering and Networks Laboratory (TIK), ETH Zurich, Zurich, Switzerland, 2001.
 */
template < class MOEOT >
class moeoSPEA2: public moeoEA < MOEOT >
{
public:

    /**
     * Ctor with a crossover, a mutation and their corresponding rates.
     * @param _maxGen number of generations before stopping
     * @param _eval evaluation function
     * @param _crossover crossover
     * @param _pCross crossover probability
     * @param _mutation mutation
     * @param _pMut mutation probability
     * @param _archive archive
     * @param _k the k-ieme distance used to fixe diversity
     * @param _nocopy boolean allow to consider copies and doublons as bad elements whose were dominated by all other MOEOT in fitness assignment.
     */
    moeoSPEA2 (unsigned int _maxGen, eoEvalFunc < MOEOT > & _eval, eoQuadOp < MOEOT > & _crossover, double _pCross, eoMonOp < MOEOT > & _mutation, double _pMut, moeoArchive < MOEOT >& _archive, unsigned int _k=1, bool _nocopy=false) :
            defaultGenContinuator(_maxGen), continuator(defaultGenContinuator), eval(_eval), loopEval(_eval), popEval(loopEval), archive(_archive),defaultSelect(2),select(defaultSelect, defaultSelect, _archive, 0.0),
            defaultSGAGenOp(_crossover, _pCross, _mutation, _pMut), fitnessAssignment(_archive, _nocopy),
            genBreed(defaultSelect, defaultSGAGenOp),selectMany(defaultSelect,0.0), selectTransform(selectMany, dummyTransform), breed(genBreed), diversityAssignment(dist,_archive, _k)
    {}


    /**
        * Ctor with a crossover, a mutation and their corresponding rates.
        * @param _continuator stopping criteria
        * @param _eval evaluation function
        * @param _crossover crossover
        * @param _pCross crossover probability
        * @param _mutation mutation
        * @param _pMut mutation probability
        * @param _archive archive
        * @param _k the k-ieme distance used to fixe diversity
        * @param _nocopy boolean allow to consider copies and doublons as bad elements whose were dominated by all other MOEOT in fitness assignment.
        */
    moeoSPEA2 (eoContinue < MOEOT >& _continuator, eoEvalFunc < MOEOT > & _eval, eoQuadOp < MOEOT > & _crossover, double _pCross, eoMonOp < MOEOT > & _mutation, double _pMut, moeoArchive < MOEOT >& _archive, unsigned int _k=1, bool _nocopy=false) :
            defaultGenContinuator(0), continuator(_continuator), eval(_eval), loopEval(_eval), popEval(loopEval), archive(_archive),defaultSelect(2),select(defaultSelect, defaultSelect, _archive, 0.0),
            defaultSGAGenOp(_crossover, _pCross, _mutation, _pMut), fitnessAssignment(_archive, _nocopy),
            genBreed(defaultSelect, defaultSGAGenOp),selectMany(defaultSelect,0.0), selectTransform(selectMany, dummyTransform), breed(genBreed), diversityAssignment(dist,_archive, _k)
    {}


    /**
        * Ctor with a crossover, a mutation and their corresponding rates.
        * @param _continuator stopping criteria
        * @param _eval pop evaluation function
        * @param _crossover crossover
        * @param _pCross crossover probability
        * @param _mutation mutation
        * @param _pMut mutation probability
        * @param _archive archive
        * @param _k the k-ieme distance used to fixe diversity
        * @param _nocopy boolean allow to consider copies and doublons as bad elements whose were dominated by all other MOEOT in fitness assignment.
        */
    moeoSPEA2 (eoContinue < MOEOT >& _continuator, eoPopEvalFunc < MOEOT > & _eval, eoQuadOp < MOEOT > & _crossover, double _pCross, eoMonOp < MOEOT > & _mutation, double _pMut, moeoArchive < MOEOT >& _archive, unsigned int _k=1, bool _nocopy=false) :
            defaultGenContinuator(0), continuator(_continuator), eval(dummyEval), loopEval(dummyEval), popEval(_eval), archive(_archive),defaultSelect(2),select(defaultSelect, defaultSelect, _archive, 0.0),
            defaultSGAGenOp(_crossover, _pCross, _mutation, _pMut), fitnessAssignment(_archive, _nocopy),
            genBreed(defaultSelect, defaultSGAGenOp),selectMany(defaultSelect,0.0), selectTransform(selectMany, dummyTransform), breed(genBreed), diversityAssignment(dist,_archive, _k)
    {}


    /**
        * Ctor with a crossover, a mutation and their corresponding rates.
        * @param _continuator stopping criteria
        * @param _eval evaluation function
        * @param _op general operator
        * @param _archive archive
        * @param _k the k-ieme distance used to fixe diversity
        * @param _nocopy boolean allow to consider copies and doublons as bad elements whose were dominated by all other MOEOT in fitness assignment.
        */
    moeoSPEA2 (eoContinue < MOEOT >& _continuator, eoEvalFunc < MOEOT > & _eval, eoGenOp < MOEOT > & _op, moeoArchive < MOEOT >& _archive, unsigned int _k=1, bool _nocopy=false) :
            defaultGenContinuator(0), continuator(_continuator), eval(_eval), loopEval(_eval), popEval(loopEval), archive(_archive),defaultSelect(2),select(defaultSelect, defaultSelect, _archive, 0.0),
            defaultSGAGenOp(defaultQuadOp, 0.0, defaultMonOp, 0.0), fitnessAssignment(_archive, _nocopy),
            genBreed(select, _op),selectMany(defaultSelect,0.0), selectTransform(selectMany, dummyTransform), breed(genBreed), diversityAssignment(dist,_archive, _k)
    {}


    /**
      * Ctor with a crossover, a mutation and their corresponding rates.
      * @param _continuator stopping criteria
      * @param _eval evaluation function
      * @param _op transformer
      * @param _archive archive
      * @param _k the k-ieme distance used to fixe diversity
      * @param _nocopy boolean allow to consider copies and doublons as bad elements whose were dominated by all other MOEOT in fitness assignment.
      */
    moeoSPEA2 (eoContinue < MOEOT >& _continuator, eoEvalFunc < MOEOT > & _eval, eoTransform < MOEOT > & _op, moeoArchive < MOEOT >& _archive, unsigned int _k=1, bool _nocopy=false) :
            defaultGenContinuator(0), continuator(_continuator), eval(_eval), loopEval(_eval), popEval(loopEval), archive(_archive),defaultSelect(2),select(defaultSelect, defaultSelect, _archive, 0.0),
            defaultSGAGenOp(defaultQuadOp, 0.0, defaultMonOp, 0.0), fitnessAssignment(_archive, _nocopy),
            genBreed(defaultSelect, defaultSGAGenOp),selectMany(select,1.0), selectTransform(selectMany, _op), breed(selectTransform), diversityAssignment(dist,_archive, _k)
    {}


    /**
      * Ctor with a crossover, a mutation and their corresponding rates.
      * @param _continuator stopping criteria
      * @param _eval pop evaluation function
      * @param _op general operator
      * @param _archive archive
      * @param _k the k-ieme distance used to fixe diversity
      * @param _nocopy boolean allow to consider copies and doublons as bad elements whose were dominated by all other MOEOT in fitness assignment.
      */
    moeoSPEA2 (eoContinue < MOEOT >& _continuator, eoPopEvalFunc < MOEOT > & _eval, eoGenOp < MOEOT > & _op, moeoArchive < MOEOT >& _archive, unsigned int _k=1, bool _nocopy=false) :
            defaultGenContinuator(0), continuator(_continuator),eval(dummyEval), loopEval(dummyEval), popEval(_eval), archive(_archive),defaultSelect(2),select(defaultSelect, defaultSelect, _archive, 0.0),
            defaultSGAGenOp(defaultQuadOp, 0.0, defaultMonOp, 0.0), fitnessAssignment(_archive, _nocopy),
            genBreed(select, _op),selectMany(defaultSelect,0.0), selectTransform(selectMany, dummyTransform), breed(genBreed), diversityAssignment(dist,_archive, _k)
    {}


    /**
      * Ctor with a crossover, a mutation and their corresponding rates.
      * @param _continuator stopping criteria
      * @param _eval pop evaluation function
      * @param _op transformer
      * @param _archive archive
      * @param _k the k-ieme distance used to fixe diversity
      * @param _nocopy boolean allow to consider copies and doublons as bad elements whose were dominated by all other MOEOT in fitness assignment.
      */
    moeoSPEA2 (eoContinue < MOEOT >& _continuator, eoPopEvalFunc < MOEOT > & _eval, eoTransform < MOEOT > & _op, moeoArchive < MOEOT >& _archive, unsigned int _k=100, bool _nocopy=false) :
            defaultGenContinuator(0), continuator(_continuator),eval(dummyEval), loopEval(dummyEval), popEval(_eval), archive(_archive),defaultSelect(2),select(defaultSelect, defaultSelect, _archive, 0.0),
            defaultSGAGenOp(defaultQuadOp, 0.0, defaultMonOp, 0.0), fitnessAssignment(_archive, _nocopy),
            genBreed(defaultSelect, defaultSGAGenOp),selectMany(select,1.0), selectTransform(selectMany, _op), breed(selectTransform), diversityAssignment(dist,_archive, _k)
    {}


    /**
     * Apply a few generation of evolution to the population _pop until the stopping criteria is verified.
     * @param _pop the population
     */
    virtual void operator () (eoPop < MOEOT > &_pop)
    {
        eoPop < MOEOT >empty_pop, offspring;
        popEval (empty_pop, _pop);// a first eval of _pop
        fitnessAssignment(_pop); //a first fitness assignment of _pop
        diversityAssignment(_pop);//a first diversity assignment of _pop
        archive(_pop);//a first filling of archive
        while (continuator (_pop))
        {
            // generate offspring, worths are recalculated if necessary
            breed (_pop, offspring);
            popEval (_pop, offspring); // eval of offspring
            // after replace, the new pop is in _pop. Worths are recalculated if necessary
            replace (_pop, offspring);
            fitnessAssignment(_pop); //fitness assignment of _pop
            diversityAssignment(_pop); //diversity assignment of _pop
            archive(_pop); //control of archive
        }
    }


protected:

    /** dummy evaluation */
	class eoDummyEval : public eoEvalFunc< MOEOT >
    {
    public:
        void operator()(MOEOT &) {}
    }
    dummyEval;

    /** dummy transform */
    class eoDummyTransform : public eoTransform<MOEOT>
    {
    public :
        void operator()(eoPop<MOEOT>&) {}
    }
    dummyTransform;

    /** a continuator based on the number of generations (used as default) */
    eoGenContinue < MOEOT > defaultGenContinuator;
    /** stopping criteria */
    eoContinue < MOEOT > & continuator;
    /** evaluation function */
    eoEvalFunc < MOEOT > & eval;
    /** loop eval */
    eoPopLoopEval < MOEOT > loopEval;
    /** evaluation function used to evaluate the whole population */
    eoPopEvalFunc < MOEOT > & popEval;
    /**archive*/
    moeoArchive < MOEOT >& archive;
    /**SelectOne*/
    moeoDetTournamentSelect < MOEOT > defaultSelect;
    /** binary tournament selection */
    moeoSelectFromPopAndArch < MOEOT > select;
    /** a default mutation */
    eoMonCloneOp < MOEOT > defaultMonOp;
    /** a default crossover */
    eoQuadCloneOp < MOEOT > defaultQuadOp;
    /** an object for genetic operators (used as default) */
    eoSGAGenOp < MOEOT > defaultSGAGenOp;
    /** fitness assignment used in NSGA-II */
    moeoDominanceCountRankingFitnessAssignment < MOEOT > fitnessAssignment;
    /** general breeder */
    eoGeneralBreeder < MOEOT > genBreed;
    /** selectMany */
    eoSelectMany <MOEOT>  selectMany;
    /** select Transform*/
    eoSelectTransform <MOEOT> selectTransform;
    /** breeder */
    eoBreed < MOEOT > & breed;
    /** diversity assignment used in NSGA-II */
    moeoNearestNeighborDiversityAssignment  < MOEOT > diversityAssignment;
    /** elitist replacement */
    moeoGenerationalReplacement < MOEOT > replace;
   /**distance*/
    moeoEuclideanDistance < MOEOT > dist;


};

#endif /*MOEOSPEA2_H_*/
