// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoGDominanceObjectiveVectorComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOGDOMINANCEOBJECTIVEVECTORCOMPARATOR_H_
#define MOEOGDOMINANCEOBJECTIVEVECTORCOMPARATOR_H_

#include <comparator/moeoObjectiveVectorComparator.h>

/**
 * This functor class allows to compare 2 objective vectors according to g-dominance.
 * The concept of g-dominance as been introduced in:
 * J. Molina, L. V. Santana, A. G. Hernandez-Diaz, C. A. Coello Coello, R. Caballero,
 * "g-dominance: Reference point based dominance" (2007)
 */
template < class ObjectiveVector >
class moeoGDominanceObjectiveVectorComparator : public moeoObjectiveVectorComparator < ObjectiveVector >
{
public:

    /**
     * Ctor.
     * @param _ref the reference point
     */
    moeoGDominanceObjectiveVectorComparator(ObjectiveVector & _ref) : ref(_ref)
    {}


    /**
     * Returns true if _objectiveVector1 is g-dominated by _objectiveVector2.
     * @param _objectiveVector1 the first objective vector
     * @param _objectiveVector2 the second objective vector
     */
    const bool operator()(const ObjectiveVector & _objectiveVector1, const ObjectiveVector & _objectiveVector2)
    {
        unsigned int flag1 = flag(_objectiveVector1);
        unsigned int flag2 = flag(_objectiveVector2);
        if (flag2==0)
        {
            // cannot dominate
            return false;
        }
        else if ( (flag2==1) && (flag1==0) )
        {
            // is dominated
            return true;
        }
        else // (flag1==1) && (flag2==1)
        {
            // both are on the good region, so let's use the classical Pareto dominance
            return paretoComparator(_objectiveVector1, _objectiveVector2);
        }
    }


private:

    /** the reference point */
    ObjectiveVector & ref;
    /** Pareto comparator */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;


    /**
     * Returns the flag of _objectiveVector according to the reference point
     * @param _objectiveVector the first objective vector
     */
    unsigned int flag(const ObjectiveVector & _objectiveVector)
    {
        unsigned int result=1;
        for (unsigned int i=0; i<ref.nObjectives(); i++)
        {
            if (_objectiveVector[i] > ref[i])
            {
                result=0;
            }
        }
        if (result==0)
        {
            result=1;
            for (unsigned int i=0; i<ref.nObjectives(); i++)
            {
                if (_objectiveVector[i] < ref[i])
                {
                    result=0;
                }
            }
        }
        return result;
    }

};

#endif /*MOEOGDOMINANCEOBJECTIVEVECTORCOMPARATOR_H_*/
