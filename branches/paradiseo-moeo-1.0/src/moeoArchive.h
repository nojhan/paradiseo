// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoArchive.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOARCHIVE_H_
#define MOEOARCHIVE_H_

#include <eoPop.h>
#include <moeoObjectiveVectorComparator.h>

/**
 * An archive is a secondary population that stores non-dominated solutions.
 */
template < class MOEOT >
class moeoArchive : public eoPop < MOEOT >
{
public:

	using std::vector < MOEOT > :: size;
	using std::vector < MOEOT > :: operator[];
	using std::vector < MOEOT > :: back;
	using std::vector < MOEOT > :: pop_back;
	
	
	/**
	 * The type of an objective vector for a solution
	 */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
	
	/**
	 * Default ctor.
	 * The moeoObjectiveVectorComparator used to compare solutions is based on Pareto dominance
	 */
	moeoArchive() : eoPop < MOEOT >(), comparator(paretoComparator)
	{}
	
	
	/**
	 * Ctor
	 * @param _comparator the moeoObjectiveVectorComparator used to compare solutions
	 */
	moeoArchive(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator) : eoPop < MOEOT >(), comparator(_comparator)
	{}
	
	
	/**
	 * Returns true if the current archive dominates _objectiveVector according to the moeoObjectiveVectorComparator given in the constructor
	 * @param _objectiveVector the objective vector to compare with the current archive
	 */
	bool dominates (const ObjectiveVector & _objectiveVector) const
	{
		for (unsigned i = 0; i<size(); i++)
		{
			if ( comparator(operator[](i).fitness(), _objectiveVector) )
			{
				return true;
			}			
		}
		return false;
	}
	
	
	/**
	 * Returns true if the current archive already contains a solution with the same objective values than _objectiveVector
	 * @param _objectiveVector the objective vector to compare with the current archive
	 */
	bool contains (const ObjectiveVector & _objectiveVector) const
	{
		for (unsigned i = 0; i<size(); i++)
		{
			if (operator[](i).fitness() == _objectiveVector)
			{
				return true;
			}
		}
		return false;
	}
	
	
	/**
	 * Updates the archive with a given individual _moeo
	 * @param _moeo the given individual
	 */
	void update (const MOEOT & _moeo)
	{
		// first step: removing the dominated solutions from the archive
		for (unsigned j=0; j<size();)
		{
			// if _moeo dominates the jth solution contained in the archive			
			if ( comparator(_moeo.objectiveVector(), operator[](j).objectiveVector()) )
			{
				operator[](j) = back();
	    		pop_back();
			}
			else if (_moeo.objectiveVector() == operator[](j).objectiveVector())
			{
				operator[](j) = back();
				pop_back();
			}
			else
			{
				j++;
			}
		}		
		// second step: is _moeo dominated?
		bool dom = false;
		for (unsigned j=0; j<size(); j++)
		{
			// if the jth solution contained in the archive dominates _moeo
			if ( comparator(operator[](j).objectiveVector(), _moeo.objectiveVector()) )
			{
				dom = true;
				break;
			}
		}
		if (!dom)
		{
			push_back(_moeo);
		}		
	}
	
	
	/**
	 * Updates the archive with a given population _pop
	 * @param _pop the given population
	 */
	void update (const eoPop < MOEOT > & _pop)
	{
		for (unsigned i=0; i<_pop.size(); i++)
		{
			update(_pop[i]);
		} 
	}
	
	
private:

	/** The moeoObjectiveVectorComparator used to compare solutions */
	moeoObjectiveVectorComparator < ObjectiveVector > & comparator;
	/** A moeoObjectiveVectorComparator	based on Pareto dominance (used as default) */
	moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;
	
};

#endif /*MOEOARCHIVE_H_ */
