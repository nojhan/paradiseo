// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoParetoObjectiveVectorComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOPARETOOBJECTIVEVECTORCOMPARATOR_H_
#define MOEOPARETOOBJECTIVEVECTORCOMPARATOR_H_

#include <comparator/moeoObjectiveVectorComparator.h>

/**
 * This functor class allows to compare 2 objective vectors according to Pareto dominance.
 */
template < class ObjectiveVector >
class moeoParetoObjectiveVectorComparator : public moeoObjectiveVectorComparator < ObjectiveVector >
{
public:

    /**
     * Returns true if _objectiveVector1 is dominated by _objectiveVector2
     * @param _objectiveVector1 the first objective vector
     * @param _objectiveVector2 the second objective vector	
     */
    const bool operator()(const ObjectiveVector & _objectiveVector1, const ObjectiveVector & _objectiveVector2)
    {
        bool dom = false;
        for (unsigned int i=0; i<ObjectiveVector::nObjectives(); i++)
        {
            // first, we have to check if the 2 objective values are not equal for the ith objective
            if ( fabs(_objectiveVector1[i] - _objectiveVector2[i]) > ObjectiveVector::Traits::tolerance() )
            {
                // if the ith objective have to be minimized...
                if (ObjectiveVector::minimizing(i))
                {
                    if (_objectiveVector1[i] > _objectiveVector2[i])
                    {
                        dom = true;		//_objectiveVector1[i] is not better than _objectiveVector2[i]
                    }
                    else
                    {
                        return false;	//_objectiveVector2 cannot dominate _objectiveVector1
                    }
                }
                // if the ith objective have to be maximized...
                else if (ObjectiveVector::maximizing(i))
                {
                    if (_objectiveVector1[i] > _objectiveVector2[i])
                    {
                        dom = true;		//_objectiveVector1[i] is not better than _objectiveVector2[i]
                    }
                    else
                    {
                        return false;	//_objectiveVector2 cannot dominate _objectiveVector1
                    }
                }
            }
        }
        return dom;
    }

};

#endif /*MOEOPARETOOBJECTIVEVECTORCOMPARATOR_H_*/
