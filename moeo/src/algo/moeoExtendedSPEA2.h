/*
  <moeoExtendedSPEA2.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------

#ifndef MOEOEXTENDEDSPEA2_H_
#define MOEOEXTENDEDSPEA2_H_

#include <eoBreed.h>
#include <eoCloneOps.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoGenContinue.h>
#include <eoGeneralBreeder.h>
#include <eoGenOp.h>
#include <eoPopEvalFunc.h>
#include <eoSGAGenOp.h>
#include <algo/moeoEA.h>
#include <GenerationalReplacement.h>
#include <DetTournamentSelect.h>
#include <selection/moeoSelectFromPopAndArch.h>

#include <diversity/moeoFuzzyNearestNeighborDiversity.h>
#include <distance/moeoBertDistance.h>
#include <archive/moeoFuzzyFixedSizeArchive.h">

/**
 * Extended SPEA2 is an extension of classical algorithm SPEA2 for incorporating the aspect of fuzziness in diversity assignment
 */
template < class MOEOT >
class moeoExtendedSPEA2: public moeoEA < MOEOT >
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
   moeoExtendedSPEA2 (unsigned int _maxGen, eoEvalFunc < MOEOT > & _eval, eoQuadOp < MOEOT > & _crossover,
   double _pCross, eoMonOp < MOEOT > & _mutation, double _pMut, moeoFuzzyArchive < MOEOT >& _archive,
   unsigned int _k=1, bool _nocopy=false) :
   defaultGenContinuator(_maxGen), continuator(defaultGenContinuator), eval(_eval), loopEval(_eval),
   popEval(loopEval), archive(_archive),defaultSelect(2),select(defaultSelect, defaultSelect, _archive, 0.0),
   defaultSGAGenOp(_crossover, _pCross, _mutation, _pMut), fitnessAssignment(_archive, _nocopy),
   genBreed(defaultSelect, defaultSGAGenOp),selectMany(defaultSelect,0.0), selectTransform(selectMany, dummyTransform),
   breed(genBreed), FuzzydiversityAssignment(dist,_archive, _k)
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

    /**SelectOne*/
    moeoDetTournamentSelect < MOEOT > defaultSelect;
    /** binary tournament selection */
    SelectFromPopAndArch < MOEOT > select;
    /** a default mutation */
    eoMonCloneOp < MOEOT > defaultMonOp;
    /** a default crossover */
    eoQuadCloneOp < MOEOT > defaultQuadOp;
    /** an object for genetic operators (used as default) */
    eoSGAGenOp < MOEOT > defaultSGAGenOp;

    /** general breeder */
    eoGeneralBreeder < MOEOT > genBreed;
    /** selectMany */
    eoSelectMany <MOEOT>  selectMany;
    /** select Transform*/
    eoSelectTransform <MOEOT> selectTransform;
    /** breeder */
    eoBreed < MOEOT > & breed;
	
	  /** Fuzzy archive*/
    moeoFuzzyArchive < MOEOT >& archive;
    /** diversity assignment used in E-SPEA2 */
    moeoFuzzyNearestNeighborDiversity < MOEOT > diversityAssignment;
    /** elitist replacement */
    moeoGenerationalReplacement < MOEOT > replace;
   /**Bert distance*/
    moeoBertDistance < MOEOT > dist;



};

#endif /*MOEOEXTENDEDSPEA2_H_*/
