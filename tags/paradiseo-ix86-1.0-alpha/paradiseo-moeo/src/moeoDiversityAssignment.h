// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoDiversityAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEODIVERSITYASSIGNMENT_H_
#define MOEODIVERSITYASSIGNMENT_H_

#include <eoFunctor.h>
#include <eoPop.h>

/**
 * Functor that sets the diversity values of a whole population.
 */
template < class MOEOT >
class moeoDiversityAssignment : public eoUF < eoPop < MOEOT > &, void >
{
public:

    /** The type for objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Updates the diversity values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    virtual void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec) = 0;


    /**
     * Updates the diversity values of the whole population _pop by taking the deletion of the individual _moeo into account.
     * @param _pop the population
     * @param _moeo the individual
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, MOEOT & _moeo)
    {
        updateByDeleting(_pop, _moeo.objectiveVector());
    }

};


/**
 * moeoDummyDiversityAssignment is a moeoDiversityAssignment that gives the value '0' as the individual's diversity for a whole population if it is invalid.
 */
template < class MOEOT >
class moeoDummyDiversityAssignment : public moeoDiversityAssignment < MOEOT >
{
public:

    /** The type for objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Sets the diversity to '0' for every individuals of the population _pop if it is invalid
     * @param _pop the population
     */
    void operator () (eoPop < MOEOT > & _pop)
    {
        for (unsigned idx = 0; idx<_pop.size (); idx++)
        {
            if (_pop[idx].invalidDiversity())
            {
                // set the diversity to 0
                _pop[idx].diversity(0.0);
            }
        }
    }


    /**
     * Updates the diversity values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        // nothing to do...  ;-)
    }

};

#endif /*MOEODIVERSITYASSIGNMENT_H_*/
