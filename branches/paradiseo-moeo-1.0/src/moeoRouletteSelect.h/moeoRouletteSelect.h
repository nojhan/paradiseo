// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoRouletteSelect.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOROULETTESELECT_H_
#define MOEOROULETTESELECT_H_

#include <moeoSelectOne.h>
#include <moeoSelectors.h>

/**
 *  moeoRouletteSelect: a selection method that selects ONE individual by
 *	using roulette wheel process
 */
template < class MOEOT >
 class moeoRouletteSelect:public moeoSelectOne <MOEOT>
{
public:
	/**
	 * Full Ctor
	 * @param _evalFitness the population fitness assignment
	 * @param _evalDiversity the population diversity assignment
	 * @param _comparator the comparator to compare the individuals
	 * @param _total 
	 */
	moeoRouletteSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoDiversityAssignment < MOEOT > &_evalDiversity, moeoComparator < MOEOT > &_comparator, double _total=1 ):evalFitness (_evalFitness), evalDiversity (_evalDiversity),
    comparator (_comparator), total (_total) {}

	/**
	 * Ctor without comparator. A moeoFitnessThenDiversityComparator is used as default.
	 * @param _evalFitness the population fitness assignment
	 * @param _evalDiversity the population diversity assignment
	 * @param _total 
	 */
	moeoRouletteSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoDiversityAssignment < MOEOT > &_evalDiversity)
		:evalFitness (_evalFitness), evalDiversity (_evalDiversity)
	    
	  {
	  	 // a moeoFitThenDivComparator is used as default
	    moeoFitnessThenDiversityComparator < MOEOT > &fitThenDivComparator;
	    comparator = fitThenDivComparator;
	  }
	  
	/**
	 * Ctor without diversity assignment. A dummy diversity assignment is used.
	 * @param _evalFitness the population fitness assignment
	 * @param _comparator the comparator to compare the individuals
	 * @param _total
	 */
moeoRouletteSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoComparator < MOEOT > &_comparator, double _total=1):evalFitness (_evalFitness), comparator (_comparator),
    total
    (_total)
  {
    // a dummy diversity is used as default
    moeoDummyDiversityAssignment < MOEOT > &dummyDiversityAssignment;
    evalDiversity = dummyDiversityAssignment;
  }


     /**
	 * Ctor without diversity assignment nor comparator. A moeoDummyDiversityAssignment and a moeoFitnessThenDiversityComparator are used as default.
	 * @param _evalFitness the population fitness assignment
	 * @param _total
	 */
moeoRouletteSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, double _total=1):evalFitness (_evalFitness),
    total (_total)
  {
    // a dummy diversity is used as default
    moeoDummyDiversityAssignment < MOEOT > &dummyDiversityAssignment;
    evalDiversity = dummyDiversityAssignment;

    // a moeoFitThenDivComparator is used as default
    moeoFitnessThenDiversityComparator < MOEOT > &fitThenDivComparator;
    comparator = fitThenDivComparator;
  }
  
	/*
	 * Evaluate the fitness and the diversity of each individual of the population.
	 */
	 void setup (eoPop<MOEOT>& _pop)
      {
      		// eval fitness
      		evalFitness(_pop);
      		
      		// eval diversity
      		evalDiversity(_pop);      
      }

  /**
   *  Apply the tournament to the given population
   */
  const MOEOT & operator  () (const eoPop < MOEOT > &_pop)
  {
  	// use the selector
   	return mo_roulette_wheel(_pop,total); //comparator ??
  }


protected:

	moeoFitnessAssignment < MOEOT > &evalFitness;

	moeoDiversityAssignment < MOEOT > &evalDiversity;

	moeoComparator < MOEOT > &comparator;

	double total;

};

#endif /*MOEOROULETTESELECT_H_ */
