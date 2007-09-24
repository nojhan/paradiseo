// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoFitnessThenDiversityComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOFITNESSTHENDIVERSITYCOMPARATOR_H_
#define MOEOFITNESSTHENDIVERSITYCOMPARATOR_H_

#include <comparator/moeoComparator.h>

/**
 * Functor allowing to compare two solutions according to their fitness values, then according to their diversity values.
 */
template < class MOEOT >
class moeoFitnessThenDiversityComparator : public moeoComparator < MOEOT >
{
public:

    /**
     * Returns true if _moeo1 < _moeo2 according to their fitness values, then according to their diversity values	
     * @param _moeo1 the first solution
     * @param _moeo2 the second solution
     */
    const bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
    {
        if (_moeo1.fitness() == _moeo2.fitness())
        {
            return _moeo1.diversity() < _moeo2.diversity();
        }
        else
        {
            return _moeo1.fitness() < _moeo2.fitness();
        }
    }

};

#endif /*MOEOFITNESSTHENDIVERSITYCOMPARATOR_H_*/
