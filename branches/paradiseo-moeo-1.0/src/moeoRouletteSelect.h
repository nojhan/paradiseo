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
 * Selection strategy that selects ONE individual by using roulette wheel process.
 */
template < class MOEOT >
 class moeoRouletteSelect:public moeoSelectOne <MOEOT>
{
public:


	/**
     * Full Ctor.
     * @param _comparator the comparator (used to compare 2 individuals)
     * @param _tSize the number of individuals in the tournament (default: 2)
     */
    moeoRouletteSelect (moeoComparator < MOEOT > &_comparator, unsigned _tSize = 2):
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
    * Ctor without comparator. A moeoFitnessThenDiversityComparator is used as default.
    * @param _tSize the number of individuals in the tournament (default: 2)	 
    */
    moeoRouletteSelect (unsigned _tSize = 2):
            comparator (*(new moeoFitnessThenDiversityComparator < MOEOT > ())),
            tSize (_tSize)
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
   *  Apply the tournament to the given population
   */
  const MOEOT & operator  () (const eoPop < MOEOT > &_pop)
  {
  	// use the selector
   	return mo_roulette_wheel(_pop,total); //comparator ??
  }


protected:

	moeoComparator < MOEOT > &comparator;

	double & total;

};

#endif /*MOEOROULETTESELECT_H_ */
