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

#include <selection/moeoSelectOne.h>
#include <selection/moeoSelectors.h>

/**
 * Selection strategy that selects ONE individual by using roulette wheel process.
 * @WARNING This selection only uses fitness values (and not diversity values).
 */
template < class MOEOT >
class moeoRouletteSelect:public moeoSelectOne < MOEOT >
{
public:

    /**
     * Ctor.
     * @param _tSize the number of individuals in the tournament (default: 2)	 
     */
    moeoRouletteSelect (unsigned int _tSize = 2) : tSize (_tSize)
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
     * Apply the tournament to the given population
     * @param _pop the population
     */
    const MOEOT & operator  () (const eoPop < MOEOT > & _pop)
    {
        // use the selector
        return mo_roulette_wheel(_pop,tSize);
    }


protected:

    /** size */
    double & tSize;

};

#endif /*MOEOROULETTESELECT_H_ */
