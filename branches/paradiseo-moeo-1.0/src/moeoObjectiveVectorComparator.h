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
class moeoObjectiveVectorComparator : public eoBF < const ObjectiveVector &, const ObjectiveVector &, int >
{};


/**
 * This functor class allows to compare 2 objective vectors according to Pareto dominance.
 */
template < class ObjectiveVector >
class moeoParetoObjectiveVectorComparator : public moeoObjectiveVectorComparator < ObjectiveVector >
{
public:
	
	/**
	 * Returns 1 if _objectiveVector1 dominates _objectiveVector2, -1 if _objectiveVector2 dominates _objectiveVector1 and 0 if no one dominates the other
	 * @param _objectiveVector1 the first objective vector
	 * @param _objectiveVector2 the second objective vector	  
	 */	 
	int operator()(const ObjectiveVector & _objectiveVector1, const ObjectiveVector & _objectiveVector2)
	{
		bool dom1 = false;
		bool dom2 = false;
		for (unsigned i=0; i<ObjectiveVector::nObjectives(); i++)
		{
			// first, we have to check if the 2 objective values are not equal for the ith objective
			if ( fabs(_objectiveVector1[i] - _objectiveVector2[i]) > ObjectiveVector::Traits::tolerance() )
			{
				// if the ith objective have to be minimized...
				if (ObjectiveVector::minimizing(i))
				{
					if (_objectiveVector1[i] > _objectiveVector2[i])
					{
						dom2 = true;	//_objectiveVector2[i] is better than _objectiveVector1[i]
					}
					else				// _objectiveVector1[i] < _objectiveVector2[i]
					{
						dom1 = true;	//_objectiveVector1[i] is better than _objectiveVector2[i]
					}
				}
				// if the ith objective have to be maximized...
				else if (ObjectiveVector::maximizing(i))
				{
					if (_objectiveVector1[i] > _objectiveVector2[i])
					{
						dom1 = true;	//_objectiveVector1[i] is better than _objectiveVector2[i]						
					}
					else				// _objectiveVector1[i] < _objectiveVector2[i]
					{
						dom2 = true;	//_objectiveVector2[i] is better than _objectiveVector1[i]
					}
				}
			}
		}
		if (dom1 == dom2)
		{
			return 0;	// no one dominates the other
		}
		if (dom1)
		{
			return 1;	//_objectiveVector1 dominates _objectiveVector2
		}
		return -1;		//_objectiveVector2 dominates _objectiveVector1
	}   

};

#endif /*MOEOOBJECTIVEVECTORCOMPARATOR_H_*/
