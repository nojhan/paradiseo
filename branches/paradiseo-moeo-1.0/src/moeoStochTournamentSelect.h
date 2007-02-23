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
 *  ???
 */
template < class MOEOT > class moeoStochTournamentSelect:public moeoSelectOne <MOEOT>
{
public:
	/**
	 * Full Ctor
	 * @param _evalFitness the population fitness assignment
	 * @param _evalDiversity the population diversity assignment
	 * @param _comparator the comparator to compare the individuals
	 * @param _tRate the tournament rate
	 */
moeoStochTournamentSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoDiversityAssignment < MOEOT > &_evalDiversity, moeoComparator < MOEOT > &_comparator, double _tRate = 1.0):evalFitness (_evalFitness), evalDiversity (_evalDiversity),
    comparator (_comparator),
    tRate (_tRate)
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
	 * @param _evalFitness the population fitness assignment
	 * @param _evalDiversity the population diversity assignment
	 * @param _tRate the tournament rate
	 */
	moeoStochTournamentSelect (moeoFitnessAssignment < MOEOT > &_evalFitness, moeoDiversityAssignment < MOEOT > &_evalDiversity)
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
	 * @param _evalFitness the population fitness assignment
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
  const MOEOT & operator() (const eoPop < MOEOT > &_pop)
  {
  	// use the selector
   	return mo_stochastic_tournament(_pop,tRate,comparator);
  }



protected:

  moeoFitnessAssignment < MOEOT > &evalFitness;

  moeoDiversityAssignment < MOEOT > &evalDiversity;

  moeoComparator < MOEOT > &comparator;

  double tRate;
};


#endif /*MOEOSTOCHTOURNAMENTSELECT_H_ */
