// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoStochTournamentSelect.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOSTOCHTOURNAMENTSELECT_H_
#define MOEOSTOCHTOURNAMENTSELECT_H_

#include <moeoSelectOne.h>
#include <moeoSelectors.h>

/**
 *  Selection strategy that selects ONE individual by stochastic tournament.
 */
template < class MOEOT > class moeoStochTournamentSelect:public moeoSelectOne <MOEOT>
{
public:

	/**
	 * Full Ctor
	 * @param _evalFitness the fitness assignment strategy
	 * @param _evalDiversity the diversity assignment strategy
	 * @param _comparator the comparator (used to compare 2 individuals)
	 * @param _tRate the tournament rate
	 */
	moeoStochTournamentSelect (moeoFitnessAssignment < MOEOT > & _evalFitness, moeoDiversityAssignment < MOEOT > & _evalDiversity, moeoComparator < MOEOT > & _comparator, double _tRate = 1.0) : 
	evalFitness (_evalFitness), evalDiversity (_evalDiversity), comparator (_comparator), tRate (_tRate)
	{
    // consistency checks
    if (tRate < 0.5)
      {
	std::
	  cerr <<
	  "Warning, Tournament rate should be > 0.5\nAdjusted to 0.55\n";
	tRate = 0.55;
      }
    if (tRate > 1)
      {
	std::
	  cerr << "Warning, Tournament rate should be < 1\nAdjusted to 1\n";
	tRate = 1;
      }
  }

	/**
	 * Ctor without comparator. A moeoFitnessThenDiversityComparator is used as default.
	 * @param _evalFitness the fitness assignment strategy
	 * @param _evalDiversity the diversity assignment strategy
	 * @param _tRate the tournament rate
	 */
	moeoStochTournamentSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoDiversityAssignment < MOEOT > &_evalDiversity, double _tRate = 1.0)
		:evalFitness (_evalFitness), evalDiversity (_evalDiversity), tRate (_tRate)
	    
	  {
	  	 // a moeoFitThenDivComparator is used as default
	    moeoFitnessThenDiversityComparator < MOEOT > &fitThenDivComparator;
	    comparator = fitThenDivComparator;
	  }
	  

	/**
	 * Ctor without diversity assignment. A dummy diversity assignment is used.
	 * @param _evalFitness the fitness assignment strategy
	 * @param _comparator the comparator (used to compare 2 individuals)
	 * @param _tRate the tournament rate
	 */
moeoStochTournamentSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoComparator < MOEOT > &_comparator, double _tRate = 1.0):evalFitness (_evalFitness), comparator (_comparator),
    tRate
    (_tRate)
  {
    // a dummy diversity is used as default
    moeoDummyDiversityAssignment < MOEOT > &dummyDiversityAssignment;
    evalDiversity = dummyDiversityAssignment;

    // consistency checks
    if (tRate < 0.5)
      {
	std::
	  cerr <<
	  "Warning, Tournament rate should be > 0.5\nAdjusted to 0.55\n";
	tRate = 0.55;
      }
    if (tRate > 1)
      {
	std::
	  cerr << "Warning, Tournament rate should be < 1\nAdjusted to 1\n";
	tRate = 1;
      }
  }


     /**
	 * Ctor without diversity assignment nor comparator. A moeoDummyDiversityAssignment and a moeoFitnessThenDiversityComparator are used as default.
	 * @param _evalFitness the fitness assignment strategy
	 * @param _tRate the tournament rate
	 */
moeoStochTournamentSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, double _tRate = 1.0):evalFitness (_evalFitness),
    tRate
    (_tRate)
  {
    // a dummy diversity is used as default
    moeoDummyDiversityAssignment < MOEOT > &dummyDiversityAssignment;
    evalDiversity = dummyDiversityAssignment;

    // a moeoFitThenDivComparator is used as default
    moeoFitnessThenDiversityComparator < MOEOT > &fitThenDivComparator;
    comparator = fitThenDivComparator;

    // consistency checks
    if (tRate < 0.5)
      {
	std::
	  cerr <<
	  "Warning, Tournament rate should be > 0.5\nAdjusted to 0.55\n";
	tRate = 0.55;
      }
    if (tRate > 1)
      {
	std::
	  cerr << "Warning, Tournament rate should be < 1\nAdjusted to 1\n";
	tRate = 1;
      }
  }


     /**
	 * Evaluate the fitness and the diversity of each individual of the population.
	 * @param _pop the population
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
   * @param _pop the population
   */
  const MOEOT & operator() (const eoPop < MOEOT > &_pop)
  {
  	// use the selector
   	return mo_stochastic_tournament(_pop,tRate,comparator);
  }



protected:
	/** the fitness assignment strategy */
	moeoFitnessAssignment < MOEOT > & evalFitness;
	/** the diversity assignment strategy */
	moeoDiversityAssignment < MOEOT > & evalDiversity;
	/** the diversity assignment strategy */
	moeoComparator < MOEOT > & comparator;
	/** the tournament rate */
	double tRate;

};

#endif /*MOEOSTOCHTOURNAMENTSELECT_H_ */
