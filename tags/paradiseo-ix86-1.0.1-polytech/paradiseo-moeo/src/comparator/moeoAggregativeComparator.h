// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoAggregativeComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOAGGREGATIVECOMPARATOR_H_
#define MOEOAGGREGATIVECOMPARATOR_H_

#include <comparator/moeoComparator.h>

/**
 * Functor allowing to compare two solutions according to their fitness and diversity values, each according to its aggregative value.
 */
template < class MOEOT >
class moeoAggregativeComparator : public moeoComparator < MOEOT >
{
public:

    /**
     * Ctor.
     * @param _weightFitness the weight for fitness
     * @param _weightDiversity the weight for diversity
     */
    moeoAggregativeComparator(double _weightFitness = 1.0, double _weightDiversity = 1.0) : weightFitness(_weightFitness), weightDiversity(_weightDiversity)
    {}


    /**
     * Returns true if _moeo1 < _moeo2 according to the aggregation of their fitness and diversity values
     * @param _moeo1 the first solution
     * @param _moeo2 the second solution
     */
    const bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
    {
        return ( weightFitness * _moeo1.fitness() + weightDiversity * _moeo1.diversity() ) < ( weightFitness * _moeo2.fitness() + weightDiversity * _moeo2.diversity() );
    }


private:

    /** the weight for fitness */
    double weightFitness;
    /** the weight for diversity */
    double weightDiversity;

};

#endif /*MOEOAGGREGATIVECOMPARATOR_H_*/
