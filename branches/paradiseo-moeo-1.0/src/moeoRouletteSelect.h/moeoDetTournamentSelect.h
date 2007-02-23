// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoDetTournamentSelect.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEODETTOURNAMENTSELECT_H_
#define MOEODETTOURNAMENTSELECT_H_

#include <moeoSelectOne.h>
#include <moeoSelectors.h>

/**
 *  moeoDetTournamentSelect: a selection method that selects ONE individual by
 *	deterministic tournament
 */
template < class MOEOT >
 class moeoDetTournamentSelect:public moeoSelectOne <MOEOT>
{
public:
	/**
	 * Full Ctor
	 * @param _evalFitness the population fitness assignment
	 * @param _evalDiversity the population diversity assignment
	 * @param _comparator the comparator to compare the individuals$
	 * @param _tSize the number of individuals in the tournament (default: 2)
	 */
	moeoDetTournamentSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoDiversityAssignment < MOEOT > &_evalDiversity, moeoComparator < MOEOT > &_comparator, unsigned _tSize = 2):evalFitness (_evalFitness), evalDiversity (_evalDiversity),
    comparator (_comparator), tSize (_tSize)
  {
    // consistency check
    if (tSize < 2)
      {
	std::
	  cout << "Warning, Tournament size should be >= 2\nAdjusted to 2\n";
	tSize = 2;
      }
  }
	
	
	     /**
	 * Ctor without comparator. A  moeoFitnessThenDiversityComparator is used as default.
	 * @param _evalFitness the population fitness assignment
	 * @param _evalDiversity the population diversity assignme
	 * @param _tSize the number of individuals in the tournament (default: 2)
	 */
	moeoDetTournamentSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoDiversityAssignment < MOEOT > &_evalDiversity, unsigned _tSize = 2)
		:evalFitness (_evalFitness),evalDiversity(_evalDiversity),tSize(_tSize)
	  {
	    // a moeoFitThenDivComparator is used as default
	    moeoFitnessThenDiversityComparator < MOEOT > &fitThenDivComparator;
	    comparator = fitThenDivComparator;
	
	    // consistency check
	    if (tSize < 2)
	      {
		std::
		  cout << "Warning, Tournament size should be >= 2\nAdjusted to 2\n";
		tSize = 2;
	      }
	  }
	  

	/**
	 * Ctor without diversity assignment. A dummy diversity assignment is used.
	 * @param _evalFitness the population fitness assignment
	 * @param _comparator the comparator to compare the individuals
	 * @param _tSize the number of individuals in the tournament (default: 2)
	 */
moeoDetTournamentSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoComparator < MOEOT > &_comparator, unsigned _tSize = 2):evalFitness (_evalFitness), comparator (_comparator),
    tSize
    (_tSize)
  {
    // a dummy diversity is used as default
    moeoDummyDiversityAssignment < MOEOT > &dummyDiversityAssignment;
    evalDiversity = dummyDiversityAssignment;

    // consistency check
    if (tSize < 2)
      {
	std::
	  cout << "Warning, Tournament size should be >= 2\nAdjusted to 2\n";
	tSize = 2;
      }
  }


     /**
	 * Ctor without diversity assignment nor comparator. A moeoDummyDiversityAssignment and a moeoFitnessThenDiversityComparator are used as default.
	 * @param _evalFitness the population fitness assignment
	 * @param _tSize the number of individuals in the tournament (default: 2)
	 */
moeoDetTournamentSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, unsigned _tSize = 2):evalFitness (_evalFitness),
    tSize
    (_tSize)
  {
    // a dummy diversity is used as default
    moeoDummyDiversityAssignment < MOEOT > &dummyDiversityAssignment;
    evalDiversity = dummyDiversityAssignment;

    // a moeoFitThenDivComparator is used as default
    moeoFitnessThenDiversityComparator < MOEOT > &fitThenDivComparator;
    comparator = fitThenDivComparator;

    // consistency check
    if (tSize < 2)
      {
	std::
	  cout << "Warning, Tournament size should be >= 2\nAdjusted to 2\n";
	tSize = 2;
      }
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
   	return mo_deterministic_tournament(_pop,tSize,comparator);
  }


protected:

	moeoFitnessAssignment < MOEOT > &evalFitness;

	moeoDiversityAssignment < MOEOT > &evalDiversity;

	moeoComparator < MOEOT > &comparator;

	unsigned tSize;

};

#endif /*MOEODETTOURNAMENTSELECT_H_ */
