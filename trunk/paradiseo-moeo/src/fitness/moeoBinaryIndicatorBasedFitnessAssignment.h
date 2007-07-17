// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoBinaryIndicatorBasedFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOBINARYINDICATORBASEDFITNESSASSIGNMENT_H_
#define MOEOBINARYINDICATORBASEDFITNESSASSIGNMENT_H_

#include <fitness/moeoIndicatorBasedFitnessAssignment.h>

/**
 * moeoIndicatorBasedFitnessAssignment for binary indicators.
 */
template < class MOEOT >
class moeoBinaryIndicatorBasedFitnessAssignment : public moeoIndicatorBasedFitnessAssignment < MOEOT >
{
public:

    /** The type for objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Updates the fitness values of the whole population _pop by taking the new objective vector _objVec into account 
     * and returns the fitness value of _objVec.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    virtual double updateByAdding(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec) = 0;

};

#endif /*MOEOINDICATORBASEDFITNESSASSIGNMENT_H_*/
