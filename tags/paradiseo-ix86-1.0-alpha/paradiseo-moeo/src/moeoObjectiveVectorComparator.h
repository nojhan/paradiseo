// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoObjectiveVectorComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOOBJECTIVEVECTORCOMPARATOR_H_
#define MOEOOBJECTIVEVECTORCOMPARATOR_H_

#include <math.h>
#include <eoFunctor.h>

/**
 * Abstract class allowing to compare 2 objective vectors.
 * The template argument ObjectiveVector have to be a moeoObjectiveVector.
 */
template < class ObjectiveVector >
class moeoObjectiveVectorComparator : public eoBF < const ObjectiveVector &, const ObjectiveVector &, bool >
    {};


/**
 * This functor class allows to compare 2 objective vectors according to Pareto dominance.
 */
template < class ObjectiveVector >
class moeoParetoObjectiveVectorComparator : public moeoObjectiveVectorComparator < ObjectiveVector >
{
public:

    /**
     * Returns true if _objectiveVector1 dominates _objectiveVector2
     * @param _objectiveVector1 the first objective vector
     * @param _objectiveVector2 the second objective vector	
     */
    bool operator()(const ObjectiveVector & _objectiveVector1, const ObjectiveVector & _objectiveVector2)
    {
        bool dom = false;
        for (unsigned i=0; i<ObjectiveVector::nObjectives(); i++)
        {
            // first, we have to check if the 2 objective values are not equal for the ith objective
            if ( fabs(_objectiveVector1[i] - _objectiveVector2[i]) > ObjectiveVector::Traits::tolerance() )
            {
                // if the ith objective have to be minimized...
                if (ObjectiveVector::minimizing(i))
                {
                    if (_objectiveVector1[i] < _objectiveVector2[i])
                    {
                        dom = true;		//_objectiveVector1[i] is better than _objectiveVector2[i]
                    }
                    else
                    {
                        return false;	//_objectiveVector1 cannot dominate _objectiveVector2
                    }
                }
                // if the ith objective have to be maximized...
                else if (ObjectiveVector::maximizing(i))
                {
                    if (_objectiveVector1[i] > _objectiveVector2[i])
                    {
                        dom = true;		//_objectiveVector1[i] is better than _objectiveVector2[i]
                    }
                    else
                    {
                        return false;	//_objectiveVector1 cannot dominate _objectiveVector2
                    }
                }
            }
        }
        return dom;
    }

};


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
    moeoGDominanceObjectiveVectorComparator(ObjectiveVector _ref) : ref(_ref)
    {}


    /**
     * Returns true if _objectiveVector1 g-dominates _objectiveVector2.
     * @param _objectiveVector1 the first objective vector
     * @param _objectiveVector2 the second objective vector
     */
    bool operator()(const ObjectiveVector & _objectiveVector1, const ObjectiveVector & _objectiveVector2)
    {
        unsigned flag1 = flag(_objectiveVector1);
        unsigned flag2 = flag(_objectiveVector2);
        if (flag1==0)
        {
            // cannot dominate
            return false;
        }
        else if ( (flag1==1) && (flag2==0) )
        {
            // dominates
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
    ObjectiveVector ref;
    /** Pareto comparator */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;


    /**
     * Returns the flag of _objectiveVector according to the reference point
     * @param _objectiveVector the first objective vector
     */
    unsigned flag(const ObjectiveVector & _objectiveVector)
    {
        unsigned result=1;
        for (unsigned i=0; i<ref.nObjectives(); i++)
        {
            if (_objectiveVector[i] > ref[i])
            {
                result=0;
            }
        }
        if (result==0)
        {
            result=1;
            for (unsigned i=0; i<ref.nObjectives(); i++)
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

#endif /*MOEOOBJECTIVEVECTORCOMPARATOR_H_*/
