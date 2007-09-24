// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOFITNESSASSIGNMENT_H_
#define MOEOFITNESSASSIGNMENT_H_

#include <eoFunctor.h>
#include <eoPop.h>

/**
 * Functor that sets the fitness values of a whole population.
 */
template < class MOEOT >
class moeoFitnessAssignment : public eoUF < eoPop < MOEOT > &, void >
{
public:

    /** The type for objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    virtual void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec) = 0;


    /**
     * Updates the fitness values of the whole population _pop by taking the deletion of the individual _moeo into account.
     * @param _pop the population
     * @param _moeo the individual
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, MOEOT & _moeo)
    {
        updateByDeleting(_pop, _moeo.objectiveVector());
    }

};

#endif /*MOEOFITNESSASSIGNMENT_H_*/
