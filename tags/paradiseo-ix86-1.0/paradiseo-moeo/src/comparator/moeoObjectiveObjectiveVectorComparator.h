// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoObjectiveObjectiveVectorComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOOBJECTIVEOBJECTIVEVECTORCOMPARATOR_H_
#define MOEOOBJECTIVEOBJECTIVEVECTORCOMPARATOR_H_

#include <comparator/moeoObjectiveVectorComparator.h>

/**
 * Functor allowing to compare two objective vectors according to their first objective value, then their second, and so on.
 */
template < class ObjectiveVector >
class moeoObjectiveObjectiveVectorComparator : public moeoObjectiveVectorComparator < ObjectiveVector >
{
public:

    /**
     * Returns true if _objectiveVector1 < _objectiveVector2 on the first objective, then on the second, and so on
     * @param _objectiveVector1 the first objective vector
     * @param _objectiveVector2 the second objective vector
     */
    const bool operator()(const ObjectiveVector & _objectiveVector1, const ObjectiveVector & _objectiveVector2)
    {
        for (unsigned int i=0; i<ObjectiveVector::nObjectives(); i++)
        {
            if ( fabs(_objectiveVector1[i] - _objectiveVector2[i]) > ObjectiveVector::Traits::tolerance() )
            {
                if (_objectiveVector1[i] < _objectiveVector2[i])
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }
        return false;
    }

};

#endif /*MOEOOBJECTIVEOBJECTIVEVECTORCOMPARATOR_H_*/
