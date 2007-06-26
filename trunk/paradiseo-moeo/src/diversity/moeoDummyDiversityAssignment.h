// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoDummyDiversityAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEODUMMYDIVERSITYASSIGNMENT_H_
#define MOEODUMMYDIVERSITYASSIGNMENT_H_

#include<diversity/moeoDiversityAssignment.h>

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
        for (unsigned int idx = 0; idx<_pop.size (); idx++)
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

#endif /*MOEODUMMYDIVERSITYASSIGNMENT_H_*/
